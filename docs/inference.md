# Inference Guide

Running style transfer locally on Apple Silicon (M1 Max 64GB).

## Setup

### 1. Quantize Base Model

The Qwen3.5-35B-A3B-Base model is 70GB in bf16 — doesn't fit in 64GB.
Quantize to 6-bit (~26GB) or 8-bit (~35GB):

```bash
pip install --upgrade mlx-lm   # Must be recent enough for qwen3_5_moe

# 6-bit (recommended — most headroom)
python -m mlx_lm convert \
    --hf-path Qwen/Qwen3.5-35B-A3B-Base \
    --mlx-path models/Qwen3.5-35B-A3B-Base-6bit-MLX \
    -q --q-bits 6
```

First run downloads ~70GB of bf16 weights. Subsequent conversions reuse the cache.

### 2. Convert PEFT Adapter to MLX

```bash
python scripts/convert_peft_to_mlx.py \
    --input lora_adapters/russell_peft \
    --output lora_adapters/russell_mlx \
    --mlx-model models/Qwen3.5-35B-A3B-Base-6bit-MLX
```

**Always pass `--mlx-model`** — without it, the adapter config points to the HuggingFace
repo ID and MLX downloads the full 70GB bf16 model at inference (instant OOM).

The script:
- Converts PEFT weight keys to MLX format
- Fixes key prefix mismatch (`model.language_model.layers.` → `language_model.model.layers.`)
- Sets the local quantized model path in `metadata.json` and `adapter_config.json`

### 3. Run

```bash
python restyle.py input.md -o output.md \
    --adapter lora_adapters/russell_mlx \
    --author "Bertrand Russell" --verbose
```

## Memory Budget

| Component | 6-bit | 8-bit |
|-----------|-------|-------|
| Base model | ~26 GB | ~35 GB |
| LoRA adapter | ~180 MB | ~180 MB |
| KV cache | ~2-4 GB | ~2-4 GB |
| OS + overhead | ~4-6 GB | ~4-6 GB |
| **Total** | **~33 GB** | **~43 GB** |
| **Headroom** | **~31 GB** | **~21 GB** |

## Configuration

Generation parameters are in `config.json` under `lora_adapters`:

```json
"lora_adapters/russell_mlx": {
    "temperature": 0.6,
    "top_p": 0.92,
    "min_p": 0.05,
    "repetition_penalty": 1.15,
    "scale": 1.0,
    "max_tokens": 512,
    "worldview": "russell_worldview.txt",
    "use_structural_rag": true,
    "logit_bias": {
        ";": -2.0,
        "—": 1.5
    }
}
```

### Per-adapter overrides

Most generation-wide settings can be overridden per-adapter. Omit the field to inherit the global `generation.*` value.

| Field | Type | Purpose |
|---|---|---|
| `expand_for_texture` | bool | Pre-expand via critic before RTT |
| `perspective` | string | Output POV (`preserve`, `first_person_singular`, …) |
| `verify_entailment` | bool | Run NLI semantic fidelity check |
| `merge_paragraphs` | int | Merge N paragraphs before LoRA |
| `use_structural_rag` | bool | Pull rhythm patterns from corpus (non-persona prompt only) |
| `chat_template` | string | LlamaFactory template the adapter was trained with (`qwen`, `qwen3_5_nothink`, `qwen3_5`, `qwen3_8`). Overrides `metadata.json`. |
| `enable_thinking` | bool | LlamaFactory `enable_thinking` used in training (default true) |
| `logit_bias` | object | Additive bias per character/string (see below) |

### `logit_bias` — per-character logit bias

Map of character/string → float. At every sampling step the value is added to that token's logit.

- **Positive** → token more likely (encourages use)
- **Negative** → token less likely (suppresses use)
- **Zero or omitted** → no effect

Magnitude controls strength. Each unit roughly shifts relative probability by a factor of `e`:

| Bias | Effect on token frequency |
|---|---|
| `+2.0` | ~7× more likely |
| `+1.5` | ~4× more likely |
| `+1.0` | ~3× more likely |
| `-1.0` | ~⅓ as often |
| `-1.5` | ~¼ as often |
| `-2.0` | ~⅐ as often ("every paragraph" → "every few paragraphs") |
| `-3.0` | Rare |
| `-5.0` | Almost never |

```json
"logit_bias": {
    "—": 1.5,      // encourage em-dashes
    ";": -2.0,     // allow semicolons occasionally, not every paragraph
    "…": -3.0      // suppress ellipses heavily
}
```

Typical range: −5.0 to +5.0. Start around ±1.5–2.0 and tune.

Resolution handles BPE quirks: each key is tokenized as both the bare string and with a leading space; if either variant is a single token, both token IDs get the bias. Keys that split into >1 token are skipped with a warning (biasing partial multi-token sequences produces artifacts).

Use this to **shape punctuation** — boost what the model underuses, suppress what it overuses — without fighting `repetition_penalty`. The two work independently: rep penalty divides logits, logit_bias adds.

### Tuning Tips

- **Style too weak**: Increase `scale` (try 3.0-4.0)
- **Incoherent output**: Decrease `temperature` (try 0.4-0.5)
- **Repeating phrases**: Increase `repetition_penalty` (try 1.2-1.3)
- **Punctuation flattened** (no em-dashes/semicolons after first use): Add a positive `logit_bias` for those characters instead of raising repetition_penalty
- **Punctuation overused** (e.g., `;` in every paragraph): Add a negative `logit_bias` (start around `-2.0`); suppression is context-aware — strong semicolon positions still fire, weak ones get filtered
- **Output too short**: This is by design (LoRA trained on ~1.21x ratio).
  Enable `expand_for_texture: true` to pre-expand content before LoRA.
- **Mechanical/robotic**: Enable `apply_input_perturbation: true` to match training distribution

## Pipeline

The full style transfer pipeline:

1. **Expand for texture** (optional): Critic model pre-expands content
2. **Perspective conversion** (optional): Convert to target POV before RTT
3. **RTT neutralization**: Strip all style via round-trip translation
4. **Input perturbation**: Add 8% noise to match training distribution
5. **LoRA generation**: Single forward pass with adapter
6. **Semantic verification**: NLI entailment check
7. **Repair** (if needed): Fix hallucinated content
8. **Post-processing**: Replace overused words, grammar correction

## Qwen 3.5 Specific Issues

### Chat Template and Thinking Tokens

The generators never use the tokenizer's own chat template (Qwen 3.5 and
Hemmingway-1 templates open a `<think>` block, Qwen 2.5's adds a different
system prompt). `render_chat_prompt` in `base_generator.py` rebuilds the exact
text LlamaFactory trained on for the adapter's template: `metadata.json`
(written by `convert_peft_to_mlx.py --train-config`) or `chat_template` in
config.json. For `qwen3_8`/`qwen3_5` with `enable_thinking: false` the prompt
ends in an empty `<think>\n\n</think>\n\n` block.

### Scale

`scale` multiplies the strength the adapter was trained at (1.0 = as trained,
0.0 = base model). Before this, MLX never applied it at all (it didn't reach
LoRA layers inside `model.layers`) and PyTorch multiplied, so MLX configs that
said 2.0 were running at 1.0. Blending several adapters stacks them into one
LoRA of combined rank, which is exactly the weighted sum of the adapters.

### Stop Tokens

The base model's EOS token is `<|endoftext|>` (248044) but chat format uses
`<|im_end|>` (248046). Without adding `<|im_end|>` to stop tokens, generation
continues into fake multi-turn conversation. This is handled in `lora_generator.py`.

### Adapter Config Model Path

Both `metadata.json` and `adapter_config.json` in the adapter directory must point
to the local quantized model path, not the HuggingFace repo ID:

```json
"base_model": "models/Qwen3.5-35B-A3B-Base-6bit-MLX"
```

Not:
```json
"base_model": "Qwen/Qwen3.5-35B-A3B-Base"   ← loads 70GB bf16, OOMs
```

The `--mlx-model` flag on `convert_peft_to_mlx.py` handles this automatically.

## Troubleshooting

- **OOM despite quantized model**: Model path in adapter config points to HF repo, not local quantized model
- **`Model type qwen3_5_moe not supported`**: `pip install --upgrade mlx-lm`
- **Output is reasoning/analysis, not restyled text**: Thinking tokens injected — check nothink template override
- **Model generates multi-turn conversation**: `<|im_end|>` not in stop tokens — check `lora_generator.py`
- **Style barely visible**: LoRA scale too low (increase in config.json) or adapter not loading (check key prefix)
- **Verify adapter loaded**: Check for `LoRALinear` modules after loading:
  ```python
  from mlx_lm import load
  model, _ = load("models/...", adapter_path="lora_adapters/...")
  lora_count = sum(1 for n, m in model.named_modules() if 'LoRA' in type(m).__name__)
  print(f"LoRA modules: {lora_count}")  # Should be ~350 for Qwen 3.5
  ```
