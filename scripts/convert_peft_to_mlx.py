#!/usr/bin/env python3
"""Convert PEFT/HuggingFace LoRA adapters to MLX format.

LLaMA-Factory produces PEFT format adapters that MLX can't load directly.
This script converts them to MLX-compatible format.

Usage:
    python scripts/convert_peft_to_mlx.py \
        --input lora_adapters/lovecraft_qwen_2.5_32b/checkpoint-600 \
        --output lora_adapters/lovecraft_32b_mlx
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def convert_peft_to_mlx(input_dir: Path, output_dir: Path):
    """Convert PEFT adapter to MLX format."""
    import safetensors.torch as st_torch
    from safetensors.numpy import save_file as save_numpy

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PEFT weights
    peft_weights_path = input_dir / "adapter_model.safetensors"
    if not peft_weights_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors found in {input_dir}")

    print(f"Loading PEFT weights from {peft_weights_path}")
    peft_weights = st_torch.load_file(str(peft_weights_path))

    # Load PEFT config
    peft_config_path = input_dir / "adapter_config.json"
    with open(peft_config_path) as f:
        peft_config = json.load(f)

    # Convert weights
    mlx_weights = {}
    for peft_key, tensor in peft_weights.items():
        # Convert key format:
        # PEFT: base_model.model.model.layers.0.mlp.down_proj.lora_A.weight
        # MLX:  model.layers.0.mlp.down_proj.lora_a

        mlx_key = peft_key

        # Remove base_model.model prefix
        if mlx_key.startswith("base_model.model."):
            mlx_key = mlx_key[len("base_model.model."):]

        # Convert lora_A.weight -> lora_a, lora_B.weight -> lora_b
        mlx_key = mlx_key.replace(".lora_A.weight", ".lora_a")
        mlx_key = mlx_key.replace(".lora_B.weight", ".lora_b")

        # Convert tensor to numpy and transpose
        # PEFT: lora_A is [rank, in_features], lora_B is [out_features, rank]
        # MLX:  lora_a is [in_features, rank], lora_b is [rank, out_features]
        np_tensor = tensor.numpy()
        np_tensor = np_tensor.T  # Transpose

        mlx_weights[mlx_key] = np_tensor

    print(f"Converted {len(mlx_weights)} weight tensors")

    # Save MLX weights
    mlx_weights_path = output_dir / "adapters.safetensors"
    save_numpy(mlx_weights, str(mlx_weights_path))
    print(f"Saved MLX weights to {mlx_weights_path}")

    # Create MLX adapter_config.json
    # MLX requires: fine_tune_type, lora_parameters, model, num_layers
    rank = peft_config.get("r", 64)
    alpha = peft_config.get("lora_alpha", 256)
    mlx_config = {
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": rank,
            "scale": alpha / rank,  # MLX scale = alpha / rank
            "dropout": peft_config.get("lora_dropout", 0.0),
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
        },
        "model": peft_config.get("base_model_name_or_path", "Qwen/Qwen2.5-32B"),
        "num_layers": -1,  # -1 means all layers
    }

    mlx_config_path = output_dir / "adapter_config.json"
    with open(mlx_config_path, "w") as f:
        json.dump(mlx_config, f, indent=4)
    print(f"Saved MLX config to {mlx_config_path}")

    # Create metadata.json
    metadata = {
        "author": "H.P. Lovecraft",
        "base_model": peft_config.get("base_model_name_or_path", "Qwen/Qwen2.5-32B"),
        "lora_rank": peft_config.get("r", 64),
        "lora_alpha": peft_config.get("lora_alpha", 256),
        "training_examples": 8840,
        "converted_from": str(input_dir),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")

    print(f"\nConversion complete! MLX adapter saved to {output_dir}")
    print(f"\nTo use:")
    print(f'  python restyle.py input.txt -o output.txt --adapter {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description="Convert PEFT/HuggingFace LoRA adapter to MLX format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to PEFT adapter directory (containing adapter_model.safetensors)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output MLX adapter directory"
    )

    args = parser.parse_args()
    convert_peft_to_mlx(args.input, args.output)


if __name__ == "__main__":
    main()
