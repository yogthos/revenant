# RunPod LoRA Training Guide

Train LoRA adapters on RunPod using LLaMA-Factory with Qwen2.5-32B.

## Requirements

- **GPU**: A100 SXM 80GB (required for Qwen2.5-32B with 4-bit quantization)
- **Template**: LLaMA-Factory
- **Estimated Time**: ~7 hours for 600 steps

---

## Step 1: Pod Setup

Create a pod using the **LLaMA-Factory template** with an A100 SXM 80GB GPU.

Once the pod is ready, install `screen` or `tmux` for long-running training:

```bash
apt update && apt install -y screen
screen -S training
```

Install required dependencies:

```bash
cd /workspace/LLaMA-Factory
pip install -e ".[metrics,bitsandbytes,qwen]"
```

---

## Step 2: Upload Training Data

Your dataset must follow SFT format:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

Upload from your local machine:

```bash
# Create directory on pod
ssh -p <PORT> root@<POD_IP> "mkdir -p /workspace/LLaMA-Factory/data/training/lovecraft"

# Upload dataset
scp -P <PORT> data/training/lovecraft/train_sft.jsonl \
    root@<POD_IP>:/workspace/LLaMA-Factory/data/training/lovecraft/train.jsonl
```

---

## Step 3: Configure Dataset

Create `dataset_info.json` on your local machine:

```json
{
  "lovecraft_sft": {
    "file_name": "training/lovecraft/train.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

Upload to pod:

```bash
scp -P <PORT> dataset_info.json root@<POD_IP>:/workspace/LLaMA-Factory/data/
```

---

## Step 4: Training Configuration

Create `qwen25_32b_lora.yaml`:

```yaml
### Model
model_name_or_path: Qwen/Qwen2.5-32B
trust_remote_code: true

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 256
lora_dropout: 0.05

### Dataset
dataset: lovecraft_sft
template: qwen
cutoff_len: 2048
packing: true
overwrite_cache: true
preprocessing_num_workers: 16

### Output
output_dir: saves/Qwen2.5-32B/lora/lovecraft
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: true

### Training
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
max_steps: 600

### Evaluation
val_size: 0.05
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100

### Optimization
quantization_bit: 4
flash_attn: sdpa
gradient_checkpointing: true
optim: paged_adamw_32bit
```

Upload to pod:

```bash
scp -P <PORT> qwen25_32b_lora.yaml root@<POD_IP>:/workspace/LLaMA-Factory/
```

---

## Step 5: Start Training

On the pod (inside screen session):

```bash
cd /workspace/LLaMA-Factory
llamafactory-cli train qwen25_32b_lora.yaml
```

Detach from screen with `Ctrl+A, D`. Reattach later with `screen -r training`.

---

## Step 6: Download Trained Adapter

Once training completes, download the adapter:

```bash
# From local machine - download the best checkpoint
scp -r -P <PORT> root@<POD_IP>:/workspace/LLaMA-Factory/saves/Qwen2.5-32B/lora/lovecraft \
    ./lora_adapters/lovecraft_peft
```

---

## Step 7: Convert to MLX Format

LLaMA-Factory produces PEFT/HuggingFace format adapters. MLX requires a different format.

### Why Conversion is Needed

| Aspect | PEFT Format | MLX Format |
|--------|-------------|------------|
| Weight keys | `base_model.model.model.layers.0...lora_A.weight` | `model.layers.0...lora_a` |
| Weight layout | `lora_A: [rank, in_features]` | `lora_a: [in_features, rank]` |
| Config fields | HuggingFace PEFT config | Requires `num_layers`, `fine_tune_type` |

### Conversion Command

```bash
# Convert the best checkpoint to MLX format
python scripts/convert_peft_to_mlx.py \
    --input lora_adapters/lovecraft_peft/checkpoint-600 \
    --output lora_adapters/lovecraft_32b_mlx
```

### Update Metadata

Edit `lora_adapters/lovecraft_32b_mlx/metadata.json` to point to your local MLX model:

```json
{
    "author": "H.P. Lovecraft",
    "base_model": "./models/Qwen2.5-32B-Base-4bit-MLX",
    "lora_rank": 64,
    "lora_alpha": 256,
    "training_examples": 8840
}
```

---

## Step 8: Prepare Base Model

The base model must be properly quantized for MLX. **Important:** Use `-q` flag:

```bash
mlx_lm.convert --hf-path Qwen/Qwen2.5-32B \
    -q --q-bits 4 \
    --mlx-path ./models/Qwen2.5-32B-Base-4bit-MLX
```

**Without `-q`**, the model saves at bf16 (~61GB instead of ~18GB).

---

## Prompt Format (Critical)

LLaMA-Factory with `template: qwen` trains models on Qwen chat format:

```
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

The inference code automatically applies this format when:
- `hasattr(tokenizer, 'apply_chat_template')` is True
- The prompt contains the `###` delimiter

If your output looks unchanged from input, verify the chat template is being applied (check logs for "Applied chat template to prompt").

---

## Training Results & Optimal Duration

Based on actual training runs, **600 steps (~3 epochs) is optimal**. Training beyond this causes overfitting:

| Step | Epoch | Train Loss | Val Loss | Notes |
|------|-------|------------|----------|-------|
| 100 | 0.48 | 1.013 | 0.977 | Early training |
| 200 | 0.96 | 0.720 | 0.703 | Learning style |
| 400 | 1.92 | 0.293 | 0.402 | Good progress |
| **600** | **2.87** | **0.131** | **0.358** | **Best checkpoint** |
| 800 | 3.83 | 0.044 | 0.392 | Overfitting begins |
| 1500 | 7.18 | 0.029 | 0.473 | Overfit |

**Key insight**: Validation loss bottoms out at step 600, then climbs 32% by step 1500. The model starts memorizing specific phrases instead of learning general style patterns.

---

## Training Time Estimates

Based on A100 80GB with the config above:

| Steps | Epochs | Time | Rate |
|-------|--------|------|------|
| 100 | 0.5 | ~1.2 hours | ~1.4 steps/min |
| 300 | 1.4 | ~3.5 hours | ~1.4 steps/min |
| **600** | **2.9** | **~7 hours** | **~1.4 steps/min** |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: bitsandbytes` | `pip install bitsandbytes>=0.39.0` |
| CUDA OOM | Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` |
| Dataset not found | Check `dataset` name matches key in `dataset_info.json` |
| Slow training | Ensure `flash_attn: sdpa` is set, check GPU utilization with `nvidia-smi` |
| Connection dropped | Use `screen` or `tmux` before starting training |
| Output too mechanical | Use earlier checkpoint (step 400-600), model may be overfit |

---

## Memory Requirements

Qwen2.5-32B with 4-bit quantization:

- **Model weights**: ~16-18GB
- **LoRA adapters (rank 64)**: ~2-3GB
- **Optimizer states**: ~6-8GB
- **Activations + gradients**: ~40-50GB
- **Total**: ~70-80GB (fits A100 80GB)

For smaller GPUs, reduce `lora_rank` to 32 or use Qwen2.5-14B instead.
