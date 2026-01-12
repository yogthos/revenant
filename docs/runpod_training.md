# RunPod LoRA Training Guide

Train LoRA adapters on RunPod using LLaMA-Factory with Qwen2.5-32B.

## Requirements

- **GPU**: A100 SXM 80GB (required for Qwen2.5-32B with 4-bit quantization)
- **Template**: LLaMA-Factory
- **Estimated Time**: ~17-18 hours for 1500 steps

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
num_train_epochs: 2.5
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
max_steps: 1500

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
# From local machine
scp -r -P <PORT> root@<POD_IP>:/workspace/LLaMA-Factory/saves/Qwen2.5-32B/lora/lovecraft \
    ./lora_adapters/lovecraft_32b
```

---

## Training Time Estimates

Based on A100 80GB with the config above:

| Progress | Steps | Time Elapsed | Rate |
|----------|-------|--------------|------|
| 5% | 75 | ~50 min | ~1.5 steps/min |
| 19% | 291 | ~3.5 hours | ~1.4 steps/min |
| 100% | 1500 | ~17-18 hours | ~1.4 steps/min |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: bitsandbytes` | `pip install bitsandbytes>=0.39.0` |
| CUDA OOM | Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` |
| Dataset not found | Check `dataset` name matches key in `dataset_info.json` |
| Slow training | Ensure `flash_attn: sdpa` is set, check GPU utilization with `nvidia-smi` |
| Connection dropped | Use `screen` or `tmux` before starting training |

---

## Memory Requirements

Qwen2.5-32B with 4-bit quantization:

- **Model weights**: ~16-18GB
- **LoRA adapters (rank 64)**: ~2-3GB
- **Optimizer states**: ~6-8GB
- **Activations + gradients**: ~40-50GB
- **Total**: ~70-80GB (fits A100 80GB)

For smaller GPUs, reduce `lora_rank` to 32 or use Qwen2.5-14B instead.
