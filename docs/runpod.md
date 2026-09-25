# RunPod Setup

Operational guide for running LoRA training on RunPod.
For training concepts and hyperparameter rationale, see `style_transfer_training.md`.

## Pod Selection

| Config | GPU | Use Case |
|--------|-----|----------|
| 1x A100 80GB | Hemmingway-1 27B QLoRA 4-bit (rank 256, ~45GB) | ~$1.64/hr |
| 1x A100 80GB | Qwen 2.5-32B QLoRA 4-bit (rank 256, ~35GB) | ~$1.64/hr |
| 2x A100 80GB | Qwen 2.5-32B bf16 + DeepSpeed ZeRO-3 (rank 256, ~40GB/GPU) | ~$3.28/hr |
| 2x H100 80GB | Qwen 3.5-35B bf16 (rank 256, ~80GB per GPU) | ~$6.58/hr |

- **Container disk**: 20GB default is fine
- **Volume disk**: 200GB+ (model weights + checkpoints — ZeRO-3 checkpoints are ~29GB each)
- **Template**: RunPod PyTorch 2.x (CUDA 12.x)

## Setup

```bash
# tmux so you can disconnect
apt update && apt install -y tmux
tmux new -s train

# attach later: tmux attach -t train

# CRITICAL: Point caches to workspace (root overlay is only 20GB)
export HF_HOME=/workspace/huggingface_cache
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets
mkdir -p $HF_DATASETS_CACHE

# Install LlamaFactory from git
pip install "llamafactory[torch] @ git+https://github.com/hiyouga/LLaMA-Factory.git"
pip install bitsandbytes

# Flash Attention 2 (optional but ~20% faster training)
pip install flash-attn --no-build-isolation

# DeepSpeed (required for 2x GPU ZeRO-3 sharding)
pip install deepspeed

# Qwen 3.5 architecture (Hemmingway-1, Qwen3.5-35B): LlamaFactory allows
# transformers <= 5.8.0, which has Qwen3_5ForCausalLM.
pip install transformers==5.8.0
# Fast Gated DeltaNet kernels. Without them transformers falls back to a slow,
# memory-hungry torch implementation of the linear-attention layers.
pip install flash-linear-attention causal-conv1d --no-build-isolation

# Clone repo
cd /workspace
git clone <your-repo-url> revenant
```

## Prepare Training Directory

### Hemmingway-1 — Russell

`train.jsonl`, `val.jsonl` and `dataset_info.json` come from
`generate_flat_training.py --format llama_factory` (or `filter_training_data.py`).
LlamaFactory reads `dataset_info.json` from `./data` by default.

```bash
mkdir -p /workspace/russell_training/data
cp revenant/data/training/russell/LlamaFactory/hemmingway1_27b_lora.yaml \
    /workspace/russell_training/
cp revenant/data/training/russell/LlamaFactory/{dataset_info.json,train.jsonl,val.jsonl} \
    /workspace/russell_training/data/
```

The yaml trains in Hemmingway-1's non-thinking chat format (`template: qwen3_8`,
`enable_thinking: false`), without packing, and keeps the checkpoint with the
lowest validation loss (`load_best_model_at_end`). The model is CC BY-NC 4.0.

### Qwen 2.5 — Howard Russell (blended)

```bash
mkdir -p /workspace/howard_russell_training/data
cp revenant/data/training/howard_russell/LlamaFactory/qwen25_32b_lora.yaml \
    /workspace/howard_russell_training/
cp revenant/data/training/howard_russell/LlamaFactory/dataset_info.json \
    /workspace/howard_russell_training/data/
cp revenant/data/training/howard_russell/LlamaFactory/train_mixed.jsonl \
    /workspace/howard_russell_training/data/
```

### Qwen 3.5 — Russell (pure)

```bash
mkdir -p /workspace/russell_training/data
cp revenant/data/training/russell/LlamaFactory/qwen35_35b_lora.yaml \
    /workspace/russell_training/
cp revenant/data/training/russell/LlamaFactory/dataset_info.json \
    /workspace/russell_training/data/
cp revenant/data/training/russell/LlamaFactory/train.jsonl \
   revenant/data/training/russell/LlamaFactory/val.jsonl \
    /workspace/russell_training/data/
```

## Train

```bash
cd /workspace/russell_training
llamafactory-cli train hemmingway1_27b_lora.yaml
# or, from howard_russell_training: llamafactory-cli train qwen25_32b_lora.yaml
```

Model auto-downloads from HuggingFace on first run.
- Hemmingway-1 27B: ~54GB (bf16, quantized to 4-bit on load)
- Qwen 2.5-32B: ~18GB (4-bit quantized during training)
- Qwen 3.5-35B-A3B: ~70GB (bf16)

## Monitor

```bash
# Adjust path for your training
tail -f saves/Qwen2.5-32B/lora/howard_russell/trainer_log.jsonl
```

First 10-20 steps: loss should be in the 1-3 range and declining. If loss spikes above
1000 or drops to 0.0, the config has a problem.

## Grabbing a Mid-Training Checkpoint

You can download and test any checkpoint while training continues. Useful for
evaluating epoch 1 quality without stopping a 3-epoch run.

```bash
# Find available checkpoints
ls saves/Qwen2.5-32B/lora/howard_russell/checkpoint-*

# Package just the adapter weights (skip optimizer state which is ~25GB)
cd saves/Qwen2.5-32B/lora/howard_russell/checkpoint-3600
tar czf /workspace/checkpoint-3600-adapter.tar.gz \
    adapter_model.safetensors adapter_config.json \
    tokenizer* chat_template* special_tokens*
```

Download locally, then convert and test:

```bash
# Convert PEFT → MLX
python scripts/convert_peft_to_mlx.py \
    --input /path/to/checkpoint-3600 \
    --output lora_adapters/howard_russell_25_32b_mlx \
    --mlx-model models/Qwen2.5-32B-Base-4bit-MLX

# Test on a chapter paragraph
python restyle.py input.md -o output.md \
    --adapter lora_adapters/howard_russell_25_32b_mlx
```

**Epoch boundaries** (with ~3,641 steps per epoch):
- Epoch 1: checkpoint-3600 or checkpoint-3700
- Epoch 2: checkpoint-7300
- Epoch 3: final (checkpoint-10900)

### Checkpoint Disk Management

DeepSpeed ZeRO-3 checkpoints are **~29GB each** (includes optimizer state shards).
With `save_steps: 100`, disk fills fast. Options:

```bash
# Option 1: Add to yaml before training
save_total_limit: 3

# Option 2: Background cleanup during training (if yaml can't be changed)
while true; do bash /workspace/revenant/scripts/cleanup_checkpoints.sh; sleep 300; done &

# Option 3: Manual cleanup
ls -d saves/Qwen2.5-32B/lora/howard_russell/checkpoint-* | sort -t- -k2 -n | head -n -3 | xargs rm -rf
```

## Run Locally (MLX)

`--train-config` records the chat template in the adapter's `metadata.json`,
so inference renders prompts exactly as training did. Without it the
generator assumes `qwen3_5_nothink`; set `chat_template` on the adapter's
config.json entry for adapters converted before this existed (`qwen` for the
Qwen 2.5 ones).

mlx_lm has no `qwen3_5_text` model type, but its `qwen3_5` module loads the
flat text config, so Hemmingway-1 needs its `model_type` changed before
conversion:

```bash
hf download Altworld/Hemmingway-1 --local-dir models/Hemmingway-1
python -c "import json; p='models/Hemmingway-1/config.json'; c=json.load(open(p)); \
    c['model_type']='qwen3_5'; json.dump(c, open(p, 'w'), indent=2)"
python -m mlx_lm convert --hf-path models/Hemmingway-1 \
    --mlx-path models/Hemmingway-1-6bit-MLX -q --q-bits 6

python scripts/convert_peft_to_mlx.py \
    --input /path/to/saves/Hemmingway-1/lora/russell \
    --output lora_adapters/russell_hemmingway_mlx \
    --mlx-model models/Hemmingway-1-6bit-MLX \
    --train-config data/training/russell/LlamaFactory/hemmingway1_27b_lora.yaml
```

The converter fails if any adapter weight doesn't match a module in the MLX
model (mlx_lm would otherwise drop it silently). `scale` in config.json
multiplies the strength the adapter was trained at, so start at 1.0.

## Upload Adapter

```bash
export HF_TOKEN=your_token_here

python -c "
import os
from huggingface_hub import HfApi, login
login(token=os.environ['HF_TOKEN'])
api = HfApi()
api.create_repo('yogthos/howard-russell-qwen25-32b-lora', private=True, exist_ok=True)
api.upload_folder(
    folder_path='saves/Qwen2.5-32B/lora/howard_russell/',
    repo_id='yogthos/howard-russell-qwen25-32b-lora',
    ignore_patterns=['checkpoint-*'],
)
print('Done!')
"
```

## Troubleshooting

**Installation:**
- **`qwen3_5` template not found**: Need LlamaFactory from git (0.9.5.dev0+), not PyPI (0.9.4). Only affects Qwen 3.5 — Qwen 2.5 uses `template: qwen` which works on any version.
- **transformers version errors**: LlamaFactory pins transformers <= 5.8.0; use 5.8.0 for the Qwen 3.5 architecture (Hemmingway-1, Qwen3.5-35B). Qwen 2.5 works with any recent version.
- **`bitsandbytes` not found**: `pip install bitsandbytes` — needed for `paged_adamw_8bit` and QLoRA.

**Disk:**
- **"No space left on device"**: Set `HF_HOME` and `HF_DATASETS_CACHE` to `/workspace/`.
- **I/O error during preprocessing**: Set `HF_DATASETS_CACHE`, reduce `preprocessing_num_workers` to 1.

**Training:**
- **"The fast path is not available" warning (Qwen 3.5)**: install `flash-linear-attention` and `causal-conv1d`; the torch fallback is slow and uses far more memory.
- **Don't turn packing on**: packed rows attend to each other and the DeltaNet layers carry state across them. LlamaFactory only isolates rows with `neat_packing` + `flash_attn: fa2` + flash-linear-attention, and fa2 has had problems with Qwen 3.5.
- **CUDA OOM**: Reduce cutoff_len → grad_accum → rank (in that order). Qwen 2.5 with QLoRA should not OOM on A100 80GB.
- **Loss spike then 0.0 (Qwen 3.5)**: rsLoRA alpha too high — see `qwen35_training.md`.
- **DDP replicates model**: Per-GPU memory = single GPU. Multi-GPU gives throughput, not more memory per card.

**Upload:**
- **`huggingface-cli` not found**: Use Python `HfApi` directly (see upload section above).
- **Upload too large**: Add `ignore_patterns=['checkpoint-*']` to skip intermediate checkpoints.
