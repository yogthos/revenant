#!/bin/bash
# One-time pod setup for Hemmingway-1 LoRA training. Everything lives on
# /workspace (the volume), so a stopped and restarted pod keeps the venv,
# model, data and checkpoints; only the container disk is wiped.
#
#   bash /workspace/revenant/scripts/runpod/setup.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN_DIR=/workspace/russell_training
export HF_HOME=/workspace/huggingface_cache

command -v tmux >/dev/null || { apt-get update && apt-get install -y tmux; }

# venv on the volume, reusing the image's torch
[ -d /workspace/venv ] || python -m venv --system-site-packages /workspace/venv
source /workspace/venv/bin/activate

pip install -U pip
pip install "llamafactory @ git+https://github.com/hiyouga/LlamaFactory.git"
# LlamaFactory allows transformers <= 5.8.0, which has Qwen3_5ForCausalLM.
pip install "transformers==5.8.0" liger-kernel flash-linear-attention
# Fast DeltaNet convolution; without it transformers falls back to torch.
pip install causal-conv1d --no-build-isolation

mkdir -p "$RUN_DIR/data"
cp "$REPO/data/training/russell/LlamaFactory/hemmingway1_27b_lora.yaml" "$RUN_DIR/"
cp "$REPO"/data/training/russell/LlamaFactory/{dataset_info.json,train.jsonl,val.jsonl} "$RUN_DIR/data/"

# ~55GB; better now than inside the first training launch.
hf download Altworld/Hemmingway-1

python - <<'PY'
import torch, fla, liger_kernel.transformers as lk, transformers
assert torch.cuda.is_available(), "no GPU"
assert hasattr(lk, "apply_liger_kernel_to_qwen3_5_text"), "liger-kernel too old for qwen3_5_text"
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
print("transformers", transformers.__version__, "| GPU", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GB")
print("DeltaNet fast path:", getattr(m, "is_fast_path_available", "unknown"))
PY
echo "Setup done. Smoke test: bash $REPO/scripts/runpod/train.sh --smoke"
