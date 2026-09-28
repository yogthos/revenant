#!/usr/bin/env python3
"""Fuse a LoRA checkpoint with the base model.

Supports three fusion paths:

1. MLX fusion (recommended for Apple Silicon):
   Loads a quantized MLX base model + MLX LoRA adapters, fuses them,
   optionally re-quantizes, and saves. No PyTorch, no OOM risk.

   python scripts/fuse_model.py \\
       --model models/Qwen2.5-32B-Base-8bit-MLX \\
       --checkpoint lora_adapters/howard_russell_checkpoint_10200 \\
       --output models/Qwen2.5-32B-howard-lovecraft-10200 \\
       --mlx --qbits 8

2. HF/PEFT fusion (PyTorch):
   Merges PEFT LoRA weights into the base model using transformers/PEFT.

   python scripts/fuse_model.py \\
       --model models/Qwen2.5-32B \\
       --checkpoint checkpoints/checkpoint-10200 \\
       --output models/style-transfer-fused

3. Convert existing HF model to MLX:
   python scripts/fuse_model.py \\
       --model models/style-transfer-fused \\
       --output models/style-transfer-mlx \\
       --convert-mlx-only
"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def fuse_mlx(
    model_path: str,
    adapter_path: Path,
    output_path: Path,
    qbits: int | None = None,
    group_size: int = 64,
    scale: float | None = None,
) -> None:
    from mlx.utils import tree_unflatten
    from mlx_lm.utils import load, quantize_model, save

    print(f"Loading MLX model: {model_path}")
    print(f"Loading MLX adapter: {adapter_path}")
    model, tokenizer, config = load(
        model_path, adapter_path=str(adapter_path), return_config=True
    )

    if scale is not None:
        print(f"Overriding LoRA scale: {scale} (was set at training time)")

    print("Fusing LoRA adapter into base model...")
    # Canonical mlx-lm fuse: call .fuse() on every LoRALinear and replace it
    # with the merged weights. `remove_lora_layers`, by contrast, returns the
    # untouched base linear and silently discards the LoRA weights.
    # If we intend to re-quantize afterwards, dequantize during fuse so
    # quantize_model sees regular Linears.
    # If --scale is provided, override the module's .scale before fusing so
    # the merged delta uses the new multiplier.
    fused_linears = []
    for n, m in model.named_modules():
        if not hasattr(m, "fuse"):
            continue
        if scale is not None:
            m.scale = scale
        fused_linears.append((n, m.fuse(dequantize=qbits is not None)))
    if not fused_linears:
        print(
            "Warning: no LoRA layers found to fuse — adapter may not be loaded correctly",
            file=sys.stderr,
        )
    else:
        model.update_modules(tree_unflatten(fused_linears))

    if qbits is not None:
        print(f"Quantizing fused model to {qbits}-bit (group_size={group_size})...")
        model, config = quantize_model(model, config, group_size=group_size, bits=qbits)

    print(f"Saving fused model to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    save(str(output_path), model_path, model, tokenizer, config)

    write_fuse_metadata(output_path, adapter_path, model_path, scale=scale, qbits=qbits, group_size=group_size)


def write_fuse_metadata(output_path: Path, adapter_path: Path, model_path: str, scale: float | None = None,
                        qbits: int | None = None, group_size: int = 64) -> None:
    """Write fuse_metadata.json, and carry the adapter's metadata.json over.

    The generator reads the training template (template, enable_thinking,
    persona_turn) from metadata.json. A fused model has no adapter, so
    without the copy its prompts fall back to the default template.
    """
    adapter_cfg = {}
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)

    trained_scale = adapter_cfg.get("lora_parameters", {}).get("scale")
    metadata = {
        "base_model": model_path,
        "adapter_path": str(adapter_path),
        "lora_rank": adapter_cfg.get("lora_parameters", {}).get("rank")
        or adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "lora_scale": scale if scale is not None else trained_scale,
        "fusion_method": "mlx",
    }
    if scale is not None and trained_scale is not None:
        metadata["lora_scale_trained"] = trained_scale
    if qbits is not None:
        metadata["quantization_bits"] = qbits
        metadata["quantization_group_size"] = group_size

    with open(output_path / "fuse_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    adapter_meta_path = adapter_path / "metadata.json"
    if adapter_meta_path.exists():
        with open(adapter_meta_path) as f:
            adapter_meta = json.load(f)
        adapter_meta["base_model"] = model_path
        with open(output_path / "metadata.json", "w") as f:
            json.dump(adapter_meta, f, indent=2)
    else:
        print("Warning: adapter has no metadata.json; set chat_template for this model in config.json",
              file=sys.stderr)


def fuse_peft(
    model_name_or_path: str, checkpoint_path: Path, output_path: Path
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    print(f"Loading PEFT adapter from {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    del base_model
    gc.collect()

    print(f"Saving fused model to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), max_shard_size="5GB")
    tokenizer.save_pretrained(str(output_path))

    del model
    gc.collect()

    with open(checkpoint_path / "adapter_config.json") as f:
        adapter_cfg = json.load(f)

    metadata = {
        "base_model": model_name_or_path,
        "checkpoint": str(checkpoint_path),
        "lora_rank": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "fusion_method": "peft",
    }
    with open(output_path / "fuse_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def convert_to_mlx(model_path: Path, output_path: Path) -> None:
    from mlx_lm.utils import load, save

    print(f"\nConverting to MLX format: {model_path} -> {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading model for MLX conversion...")
    model, tokenizer, config = load(str(model_path), return_config=True)

    save(str(output_path), str(model_path), model, tokenizer, config)

    if (model_path / "fuse_metadata.json").exists():
        with open(model_path / "fuse_metadata.json") as f:
            metadata = json.load(f)
        metadata["mlx_converted"] = True
        with open(output_path / "fuse_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    # The training template the generator reads.
    if (model_path / "metadata.json").exists():
        shutil.copy(model_path / "metadata.json", output_path / "metadata.json")

    print(f"MLX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA checkpoint with base model")
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="HuggingFace model name or local path to base model",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Path to PEFT checkpoint or MLX adapter directory to fuse",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for the fused model",
    )
    parser.add_argument(
        "--mlx",
        action="store_true",
        help="Use MLX for fusion (requires MLX-format base model and adapters)",
    )
    parser.add_argument(
        "--qbits",
        type=int,
        default=None,
        help="Quantize output to N bits after MLX fusion (e.g. 8 for Q8)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override the LoRA scale at fuse time (e.g. 0.5 weakens the "
        "adapter, 3.0 strengthens it). Defaults to the scale from the "
        "adapter's config.json.",
    )
    parser.add_argument(
        "--convert-mlx",
        action="store_true",
        help="Convert the fused model to MLX format after fusing",
    )
    parser.add_argument(
        "--convert-mlx-only",
        action="store_true",
        help="Skip fusion; just convert an existing HF model to MLX",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.convert_mlx_only:
        convert_to_mlx(Path(args.model), output_path)
        print(f"\nDone! MLX model saved to {output_path}")
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required when not using --convert-mlx-only")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    if args.mlx:
        fuse_mlx(
            args.model,
            checkpoint_path,
            output_path,
            qbits=args.qbits,
            group_size=args.group_size,
            scale=args.scale,
        )
    else:
        if args.scale is not None:
            print(
                "Error: --scale is only supported with --mlx. PEFT fusion applies "
                "the adapter's trained alpha/r scaling via merge_and_unload().",
                file=sys.stderr,
            )
            sys.exit(1)
        fuse_peft(args.model, checkpoint_path, output_path)
        if args.convert_mlx:
            mlx_output = output_path.parent / (output_path.name + "-MLX")
            convert_to_mlx(output_path, mlx_output)
            print(f"\nMLX model saved to {mlx_output}")

    print(f"\nFused model saved to {output_path}")
    print(f"\nUsage:")
    print(f"  python restyle.py input.txt -o output.txt --model {output_path}")


if __name__ == "__main__":
    main()
