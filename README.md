# Revenant

Transform text to match a target author's writing style while preserving semantic meaning. Uses LoRA-adapted language models for fast, consistent style transfer.

## Requirements

- Python 3.9+
- Apple Silicon Mac (for MLX inference)
- ~30GB disk space (6-bit quantized model + adapter)
- DeepSeek API key (for RTT neutralization and training data generation)

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
cp config.json.sample config.json   # Add your DeepSeek API key

# Download and quantize base model (one-time, ~70GB download → 26GB output)
python -m mlx_lm convert \
    --hf-path Qwen/Qwen3.5-35B-A3B-Base \
    --mlx-path models/Qwen3.5-35B-A3B-Base-6bit-MLX \
    -q --q-bits 6

# Convert adapter (if using PEFT format from cloud training)
python scripts/convert_peft_to_mlx.py \
    --input lora_adapters/author_peft \
    --output lora_adapters/author_mlx \
    --mlx-model models/Qwen3.5-35B-A3B-Base-6bit-MLX \
    --train-config data/training/author/LlamaFactory/qwen35_35b_lora.yaml

# Run style transfer
python restyle.py input.md -o output.md \
    --adapter lora_adapters/author_mlx \
    --author "Author Name"
```

## Usage

```bash
# Basic transfer
python restyle.py input.md -o output.md --author "Bertrand Russell"

# With adapter scale override
python restyle.py input.md -o output.md \
    --adapter lora_adapters/russell_mlx:2.0 \
    --author "Bertrand Russell"

# Interactive mode (5 variations per input)
python restyle.py --repl --adapter lora_adapters/russell_mlx --author "Bertrand Russell"

# Skip verification for speed
python restyle.py input.md -o output.md --no-verify

# List available adapters
python restyle.py --list-adapters
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--adapter PATH[:SCALE]` | config | LoRA adapter path with optional scale (can repeat to blend) |
| `--model PATH` | config | Fused model path (LoRA pre-merged into base); overrides `--adapter`. See [docs/fused_models.md](docs/fused_models.md) |
| `--author NAME` | - | Author name (optional if adapter has metadata) |
| `--temperature FLOAT` | config | Generation temperature |
| `--perspective MODE` | config | Output perspective: `preserve`, `first_person_singular`, `first_person_plural`, `third_person`, `author_voice_third_person` |
| `--lora-scale FLOAT` | config | LoRA influence scale |
| `--expand` / `--no-expand` | config | Enable/disable texture expansion |
| `--no-verify` | false | Skip entailment verification |
| `--repl` | false | Interactive mode |
| `--list-adapters` | false | List available LoRA adapters |
| `-v` | false | Verbose output |

## Architecture

```
Input → Perspective Conversion → RTT Neutralization → Perturbation → LoRA Generation → Verification → Output
```

| Stage | Purpose |
|-------|---------|
| Perspective Conversion | Convert to target POV before neutralization |
| RTT Neutralization | Strip source style via English → Mandarin → English |
| Perturbation | Add 8% noise to match training distribution |
| LoRA Generation | Single forward pass with style adapter |
| Semantic Verification | NLI entailment + entity checking |
| Post-Processing | Replace overused words, grammar correction |

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/generation/lora_generator.py` | MLX LoRA inference |
| `src/generation/transfer.py` | Pipeline orchestration |
| `src/validation/semantic_verifier.py` | Content verification |
| `src/vocabulary/repetition_reducer.py` | Post-processing |
| `src/persona/prompt_builder.py` | Author persona prompts |
| `src/rag/structural_rag.py` | Style pattern retrieval |

## Training Your Own Adapter

### Overview

```
1. Curate Corpus → 2. Generate Training Data → 3. Train LoRA → 4. Convert to MLX
```

### 1. Curate Corpus

Collect 50k+ words of clean author prose:

```bash
python scripts/curate_corpus.py \
    --input data/corpus/raw/author.txt \
    --output data/corpus/curated/author.txt
```

Then clean it. Anything left in the corpus (Gutenberg markup, footnote marks, stripped
formulas, OCR errors) ends up in training targets:

```bash
python scripts/clean_corpus.py data/corpus/curated/author.txt
```

A badly scanned book can be rebuilt from its raw OCR text, with DeepSeek fixing spelling and
OCR errors only (replies that reword are rejected; results are cached next to the raw file):

```bash
python scripts/clean_corpus.py data/corpus/curated/author.txt \
    --rebuild data/corpus/author/scanned_book.txt --ocr-fix
```

### 2. Generate Training Data

```bash
python scripts/generate_flat_training.py \
    --corpus data/corpus/curated/author.txt \
    --author "Author Name" \
    --output data/training/author \
    --snowflake-topics data/training/author/snowflake_topics.py \
    --format llama_factory --skip-curation --workers 4
```

This ends by filtering rows and writing `LlamaFactory/train.jsonl`, `val.jsonl` (held out by
source paragraph) and `dataset_info.json`. To rerun that step on its own:

```bash
python scripts/filter_training_data.py data/training/author/train.jsonl
```

### 3. Train LoRA

Training requires a GPU with 80GB+ VRAM (A100 or H100 on RunPod).
See [docs/runpod.md](docs/runpod.md) for cloud training setup. The Russell
config for Altworld/Hemmingway-1 is
`data/training/russell/LlamaFactory/hemmingway1_27b_lora.yaml`.

### 4. Convert to MLX

```bash
python scripts/convert_peft_to_mlx.py \
    --input lora_adapters/author_peft \
    --output lora_adapters/author_mlx \
    --mlx-model models/Qwen3.5-35B-A3B-Base-6bit-MLX \
    --train-config data/training/author/LlamaFactory/qwen35_35b_lora.yaml
```

See [docs/inference.md](docs/inference.md) for detailed inference setup.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/style_transfer_training.md](docs/style_transfer_training.md) | Training principles, data format, hyperparameters, experimental findings |
| [docs/qwen25_training.md](docs/qwen25_training.md) | Qwen 2.5 dense model specifics |
| [docs/qwen35_training.md](docs/qwen35_training.md) | Qwen 3.5 MoE specifics, rsLoRA, config evolution |
| [docs/inference.md](docs/inference.md) | Local MLX setup, adapter conversion, pipeline |
| [docs/runpod.md](docs/runpod.md) | RunPod cloud training setup |

## Configuration

Key settings in `config.json`:

```json
{
  "generation": {
    "expand_for_texture": false,
    "apply_input_perturbation": true,
    "use_structural_rag": true,
    "use_persona": true,
    "lora_adapters": {
      "lora_adapters/russell_mlx": {
        "enabled": true,
        "scale": 2.0,
        "temperature": 0.7,
        "worldview": "russell_worldview.txt",
        "logit_bias": { ";": -2.0, "—": 1.5 }
      }
    }
  }
}
```

| Setting | Description |
|---------|-------------|
| `scale` | LoRA influence (0.0=base, 1.0=full, >1.0=amplified) |
| `temperature` | Generation creativity (lower=more coherent) |
| `worldview` | Persona prompt file in `prompts/` (must match training frames exactly) |
| `apply_input_perturbation` | Add 8% noise to match training distribution |
| `expand_for_texture` | Pre-expand content for longer output |
| `logit_bias` | Per-character additive bias. Positive = encourage (e.g. `"—": 1.5`), negative = suppress (e.g. `";": -2.0`). Typical range −5 to +5; start around ±1.5–2.0 and tune |
| `use_structural_rag` | Per-adapter override for the global RAG flag |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Style too weak | Increase adapter `scale` in config.json |
| Output unchanged | Adapter not loading — check `metadata.json` points to local model |
| Content hallucinated | Lower `scale`, check semantic verification is enabled |
| OOM at inference | Model path points to HF repo (70GB) not local quantized model |
| "Empty content in DeepSeek response" | `deepseek-v4-flash` runs in thinking mode by default; the reasoning chain can exhaust a small `max_tokens` budget before any content is emitted. Set `thinking: false` on the provider in config.json. Note: thinking mode also ignores `temperature`/`top_p` |
| Thinking tokens in output | Chat template override missing — see [docs/inference.md](docs/inference.md) |
| Multi-turn conversation | `<|im_end|>` not in stop tokens — see [docs/inference.md](docs/inference.md) |

## License

MIT License
