#!/usr/bin/env python3
"""Convert PEFT/HuggingFace LoRA adapters to MLX format.

LLaMA-Factory produces PEFT format adapters that MLX can't load directly.
This script converts them to MLX-compatible format.

Usage:
    python scripts/convert_peft_to_mlx.py \
        --input saves/Hemmingway-1/lora/russell \
        --output lora_adapters/russell_hemmingway_mlx \
        --mlx-model models/Hemmingway-1-6bit-MLX \
        --train-config data/training/russell/LlamaFactory/hemmingway1_27b_lora.yaml

--train-config records the LlamaFactory template in metadata.json so inference
renders prompts exactly as training did.
"""

import argparse
import json
from pathlib import Path


def _detect_model_prefix(input_dir: Path, peft_weights: dict) -> str:
    """Detect the correct layer key prefix by checking the MLX base model.

    Different architectures use different prefixes:
      Qwen 2.5:     model.layers.N...
      Qwen 3.5 MoE: language_model.model.layers.N...

    Returns the prefix string up to and including "layers.N." or empty string
    if detection fails.
    """
    import re

    # Find PEFT adapter_config.json in the input directory
    peft_config_path = input_dir / "adapter_config.json"
    if not peft_config_path.exists():
        print(f"  No adapter_config.json in {input_dir}, skipping prefix detection")
        return ""

    # Look at a PEFT key to see what prefix it has after stripping base_model.model.
    sample_peft_key = next(iter(peft_weights))
    stripped = sample_peft_key
    if stripped.startswith("base_model.model."):
        stripped = stripped[len("base_model.model."):]

    # Extract everything before "layers.N."
    m = re.match(r'(.+?layers\.)\d+\.', stripped)
    peft_prefix = m.group(1) if m else ""

    # Now check what prefix the MLX model actually uses
    # Look for safetensors files in common model locations
    base_model_path = None
    if peft_config_path.exists():
        with open(peft_config_path) as f:
            cfg = json.load(f)
        candidate = cfg.get("base_model_name_or_path", cfg.get("model", ""))
        if candidate and Path(candidate).exists():
            base_model_path = Path(candidate)

    if base_model_path is None:
        print(f"  Could not find base model to detect prefix, using PEFT prefix: {peft_prefix}")
        return ""

    # Read a safetensors file from the base model to get actual key names
    st_files = sorted(base_model_path.glob("*.safetensors"))
    if not st_files:
        return ""

    from safetensors import safe_open
    with safe_open(str(st_files[0]), framework="numpy") as f:
        model_keys = f.keys()
        # Find a key with "layers.0."
        for mk in model_keys:
            m2 = re.match(r'(.+?layers\.)\d+\.', mk)
            if m2:
                mlx_prefix = m2.group(1)
                if mlx_prefix != peft_prefix:
                    print(f"  Prefix mismatch: PEFT='{peft_prefix}' MLX='{mlx_prefix}'")
                    return mlx_prefix
                else:
                    print(f"  Prefix matches: '{peft_prefix}'")
                    return ""

    return ""


def read_train_config(path) -> dict:
    """Prompt layout settings from a LlamaFactory training yaml.

    persona_turn comes from the dataset_info.json next to the yaml: rows with
    a system column put the persona in the system turn, alpaca rows without
    one put it in the user turn with the input.
    """
    import yaml
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    persona_turn = "user"
    info_path = path.parent / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        dataset = str(cfg["dataset"]).split(",")[0].strip()
        if "system" in info.get(dataset, {}).get("columns", {}):
            persona_turn = "system"
    else:
        print(f"  No {info_path}; assuming the persona was in the user turn")
    # LlamaFactory's enable_thinking defaults to true.
    return {"template": cfg["template"], "enable_thinking": cfg.get("enable_thinking", True),
            "persona_turn": persona_turn}


def build_mlx_model(model_path: Path):
    """The model mlx_lm builds from ``model_path``'s config, without weights.

    Its parameter names are the ones adapters must use. They can differ from
    the names in the model's files: a Hugging Face Qwen3.5 folder says
    model.layers.N..., mlx_lm's module is language_model.model.layers.N....
    """
    from mlx_lm.utils import _get_classes, load_config

    config = load_config(Path(model_path))
    model_class, args_class = _get_classes(config=config)
    return model_class(args_class.from_dict(config))


def mlx_module_names(model_path: Path) -> set:
    from mlx.utils import tree_flatten

    return {k[: -len(".weight")] for k, _ in tree_flatten(build_mlx_model(model_path).parameters())
            if k.endswith(".weight")}


def rename_to_model(mlx_weights: dict, model_path: Path) -> dict:
    """Give every LoRA weight the name of the module mlx_lm builds for it.

    Matching is on the "layers.N.module" tail, so any wrapper prefix works.
    mlx_lm loads adapters with strict=False: a name that matches no module is
    dropped silently and that part of the adapter never runs, so a tail with
    no module is an error.
    """
    import re

    by_tail = {}
    for name in mlx_module_names(model_path):
        m = re.search(r"layers\.\d+\..+$", name)
        if m:
            by_tail[m.group(0)] = name
    renamed, missing = {}, set()
    for key, value in mlx_weights.items():
        module, _, part = key.rpartition(".lora_")
        m = re.search(r"layers\.\d+\..+$", module)
        target = by_tail.get(m.group(0)) if m else None
        if target is None:
            missing.add(module)
            continue
        renamed[f"{target}.lora_{part}"] = value
    if missing:
        raise ValueError(f"{len(missing)} adapted modules are not in {model_path}, e.g. {sorted(missing)[:3]}")
    return renamed


def convert_peft_to_mlx(input_dir: Path, output_dir: Path, mlx_model_path: str = None, author: str = None,
                        train_config=None):
    """Convert PEFT adapter to MLX format."""
    import safetensors.torch as st_torch
    from safetensors.numpy import save_file as save_numpy

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

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
    # Need to map PEFT key prefixes to MLX key prefixes.
    # PEFT format: base_model.model.{mlx_model_prefix}.layers.N.module.lora_A.weight
    # MLX format:  {mlx_model_prefix}.layers.N.module.lora_a
    #
    # The mlx_model_prefix varies by architecture:
    #   Qwen 2.5:     model.layers.N...
    #   Qwen 3.5 MoE: language_model.model.layers.N...
    #
    # We detect the prefix by looking at the actual model weight names.
    model_prefix = ""
    if mlx_model_path:
        if not (Path(mlx_model_path) / "config.json").exists():
            raise FileNotFoundError(f"--mlx-model {mlx_model_path} has no config.json; the adapter's "
                                    "weight names can't be checked against it")
    else:
        model_prefix = _detect_model_prefix(input_dir, peft_weights)

    mlx_weights = {}
    for peft_key, tensor in peft_weights.items():
        mlx_key = peft_key

        # Remove base_model.model. prefix (PEFT wrapper)
        if mlx_key.startswith("base_model.model."):
            mlx_key = mlx_key[len("base_model.model."):]

        # If we detected a model prefix mismatch, fix it
        if model_prefix:
            # The PEFT key after stripping base_model.model. starts with the
            # HF model's internal prefix. We need to match the MLX model's prefix.
            # e.g. PEFT: "model.language_model.layers.0..." -> MLX: "language_model.model.layers.0..."
            pass  # model_prefix fixup is done below after lora_A/B conversion

        # Convert lora_A.weight -> lora_a, lora_B.weight -> lora_b
        mlx_key = mlx_key.replace(".lora_A.weight", ".lora_a")
        mlx_key = mlx_key.replace(".lora_B.weight", ".lora_b")

        # Convert tensor to numpy and transpose
        # PEFT: lora_A is [rank, in_features], lora_B is [out_features, rank]
        # MLX:  lora_a is [in_features, rank], lora_b is [rank, out_features]
        # safetensors.torch returns torch tensors; bf16 can't go to numpy directly
        np_tensor = tensor.float().numpy().T  # bf16→f32→numpy, then transpose

        mlx_weights[mlx_key] = np_tensor

    # Fix key prefixes to match the MLX model's actual weight names
    # model_prefix is e.g. "language_model.model.layers." (includes trailing "layers.")
    # We need to replace everything before "layers.N." with the model_prefix
    if model_prefix:
        import re
        fixed_weights = {}
        for key, val in mlx_weights.items():
            # Replace everything up to and including the first "layers." with model_prefix
            # e.g. "model.language_model.layers.0.foo" -> "language_model.model.layers.0.foo"
            fixed_key = re.sub(r'^.*?layers\.', model_prefix, key, count=1)
            fixed_weights[fixed_key] = val
        mlx_weights = fixed_weights
        print(f"Fixed key prefix to: {model_prefix}")

    print(f"Converted {len(mlx_weights)} weight tensors")

    if mlx_model_path:
        mlx_weights = rename_to_model(mlx_weights, Path(mlx_model_path))
        print(f"Weight names match the modules mlx_lm builds from {mlx_model_path}")

    # Save MLX weights
    output_dir.mkdir(parents=True, exist_ok=True)
    mlx_weights_path = output_dir / "adapters.safetensors"
    save_numpy(mlx_weights, str(mlx_weights_path))
    print(f"Saved MLX weights to {mlx_weights_path}")

    # Create MLX adapter_config.json
    # MLX requires: fine_tune_type, lora_parameters, model, num_layers
    rank = peft_config.get("r", 64)
    alpha = peft_config.get("lora_alpha", 256)

    # Auto-detect LoRA target keys from the converted weight names
    # e.g. "language_model.model.layers.0.self_attn.q_proj.lora_a" -> "self_attn.q_proj"
    import re as _re
    lora_keys = set()
    for key in mlx_weights:
        if ".lora_a" in key or ".lora_b" in key:
            parts = key.replace(".lora_a", "").replace(".lora_b", "")
            # Strip everything up to and including "layers.N."
            match = _re.sub(r"^.*?layers\.\d+\.", "", parts)
            if match != parts:  # successfully stripped
                lora_keys.add(match)

    # Fall back to PEFT target_modules if auto-detect fails
    if not lora_keys:
        lora_keys = set(peft_config.get("target_modules", [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ]))

    print(f"Detected LoRA keys: {sorted(lora_keys)}")

    mlx_config = {
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": rank,
            # MLX doesn't support rsLoRA — it applies scale directly as a multiplier.
            # If rsLoRA was used in training, effective scale = alpha/sqrt(rank).
            # If standard LoRA, effective scale = alpha/rank.
            "scale": alpha / (rank ** 0.5) if peft_config.get("use_rslora", False) else alpha / rank,
            "dropout": peft_config.get("lora_dropout", 0.0),
            "keys": sorted(lora_keys),
        },
        # Use local MLX model path if provided, otherwise fall back to PEFT config
        "model": mlx_model_path or peft_config.get("base_model_name_or_path", "Qwen/Qwen2.5-32B"),
        "num_layers": -1,  # -1 means all layers
    }

    mlx_config_path = output_dir / "adapter_config.json"
    with open(mlx_config_path, "w") as f:
        json.dump(mlx_config, f, indent=4)
    print(f"Saved MLX config to {mlx_config_path}")

    # Create metadata.json
    base_model_ref = mlx_model_path or peft_config.get("base_model_name_or_path", "Qwen/Qwen2.5-32B")
    metadata = {
        "author": author or "Unknown",
        "base_model": base_model_ref,
        "lora_rank": peft_config.get("r", 64),
        "lora_alpha": peft_config.get("lora_alpha", 256),
        "converted_from": str(input_dir),
    }
    if train_config:
        metadata.update(read_train_config(train_config))
    else:
        print("  No --train-config: inference will assume the qwen3_5_nothink template")

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")

    print(f"\nConversion complete! MLX adapter saved to {output_dir}")
    print("\nTo use:")
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
    parser.add_argument(
        "--mlx-model",
        required=False,
        help="Path to local MLX base model (for prefix detection and config). "
             "E.g., models/Qwen3.5-35B-A3B-Base-6bit-MLX"
    )
    parser.add_argument(
        "--train-config",
        required=False,
        help="LlamaFactory yaml the adapter was trained with (records its chat template)",
    )
    parser.add_argument(
        "--author",
        required=False,
        help="Author name for adapter metadata (e.g., 'Howard Russell')"
    )

    args = parser.parse_args()
    convert_peft_to_mlx(args.input, args.output, mlx_model_path=args.mlx_model, author=args.author,
                        train_config=args.train_config)


if __name__ == "__main__":
    main()
