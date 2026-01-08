# Text Style Transfer

Transform text to match a target author's writing style while preserving semantic meaning. Uses LoRA-adapted language models for fast, consistent style transfer.

## Getting Started

### Using a Pre-trained Adapter

A pre-trained H.P. Lovecraft adapter is available on HuggingFace: [yogthos/lovecraft-style-lora-14b](https://huggingface.co/yogthos/lovecraft-style-lora-14b)

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Download the pre-trained adapter
mkdir -p lora_adapters
cd lora_adapters
git lfs install
git clone https://huggingface.co/yogthos/lovecraft-style-lora-14b lovecraft
cd ..

# Copy config and add your DeepSeek API key
cp config.json.sample config.json

# Run style transfer
python restyle.py input.txt -o output.txt --author "H.P. Lovecraft"
```

### Requirements

- Python 3.9+
- Apple Silicon Mac (for MLX inference)
- DeepSeek API key (for RTT neutralization)

## Usage

```bash
# Basic transfer
python restyle.py input.txt -o output.txt --author "H.P. Lovecraft"

# With adapter scale (0.0-1.0)
python restyle.py input.txt -o output.txt \
    --adapter lora_adapters/lovecraft:0.3 \
    --author "H.P. Lovecraft"

# Interactive mode (5 variations per input)
python restyle.py --repl --adapter lora_adapters/lovecraft --author "H.P. Lovecraft"

# Skip verification for speed
python restyle.py input.txt -o output.txt --no-verify
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--adapter PATH[:SCALE]` | - | LoRA adapter path with optional scale |
| `--author NAME` | - | Author name |
| `--checkpoint FILE` | - | Use specific training checkpoint |
| `--temperature` | 0.4 | Generation temperature |
| `--no-verify` | false | Skip entailment verification |
| `--repl` | false | Interactive mode |
| `-v` | false | Verbose output |

---

## Training Your Own Adapter

### Pipeline Overview

```
1. Curate Corpus → 2. Generate Training Data → 3. Index ChromaDB → 4. Train LoRA → 5. Configure Persona
```

### Step 1: Curate Corpus

```bash
python scripts/curate_corpus.py \
    --input data/corpus/raw/author.txt \
    --output data/corpus/author.txt \
    --target-tokens 900000
```

Filters low-quality text, selects diverse passages, caps at target token budget.

### Step 2: Generate Training Data

```bash
python scripts/generate_flat_training.py \
    --corpus data/corpus/author.txt \
    --author "Author Name" \
    --output data/training/author
```

Creates training pairs via RTT neutralization: `neutral_text → styled_text`.

### Step 3: Index Corpus

```bash
python scripts/load_corpus.py \
    --input data/corpus/author.txt \
    --author "Author Name"
```

Indexes corpus in ChromaDB for Structural RAG and Grafting during inference.

### Step 4: Train LoRA

Create `data/training/author/config.yaml`:

```yaml
model: "./models/Qwen2.5-14B-Base-4bit-MLX"
train: true
data: "data/training/author"

batch_size: 1
grad_accumulation: 4
grad_checkpoint: true
iters: 2100
learning_rate: 1e-5

lora_parameters:
  rank: 64
  scale: 2.0
  dropout: 0.1
  keys:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"
    - "mlp.gate_proj"
    - "mlp.up_proj"
    - "mlp.down_proj"

adapter_path: "lora_adapters/author"
save_every: 200
```

Run training:

```bash
mlx_lm.lora --config data/training/author/config.yaml
```

Create `lora_adapters/author/metadata.json`:

```json
{
    "author": "Author Name",
    "base_model": "./models/Qwen2.5-14B-Base-4bit-MLX",
    "lora_rank": 64,
    "lora_alpha": 128,
    "training_examples": 3770
}
```

### Step 5: Configure Persona

Create `prompts/author_persona.txt` with persona frames matching training:

```
[PERSONA_FRAMES_NARRATIVE]
{Narrative frame from training}
---
{Another narrative frame}

[PERSONA_FRAMES_CONCEPTUAL]
{Conceptual frame from training}
---
{Another conceptual frame}
```

Update `config.json`:

```json
"lora": {
  "worldview": "author_persona.txt"
}
```

---

## Base Model Setup

Download and quantize a base model:

```bash
mkdir -p models
mlx_lm.convert --hf-path Qwen/Qwen2.5-14B \
    --q-bits 4 \
    --mlx-path models/Qwen2.5-14B-Base-4bit-MLX
```

| Model | Training Memory | Inference Memory |
|-------|-----------------|------------------|
| Qwen2.5-7B-4bit | ~20GB | ~8GB |
| **Qwen2.5-14B-4bit** | ~40GB | ~16GB |
| Qwen2.5-32B-4bit | ~64GB+ | ~24GB |

---

## Architecture

### Pipeline

```
Input → RTT Neutralization → Structural RAG → LoRA Generation → Semantic Verification → Output
```

- **RTT Neutralization**: Strips source style via English → Mandarin → English
- **Structural RAG**: Retrieves rhythm patterns from corpus
- **Structural Grafting**: Copies rhetorical skeleton from similar passages
- **Semantic Verification**: NLI entailment + entity checking

---

## Configuration

Key settings in `config.json`:

```json
{
  "generation": {
    "entailment_threshold": 0.9,
    "lora_adapters": {
      "lora_adapters/lovecraft": {"scale": 0.3}
    },
    "use_structural_rag": true,
    "use_structural_grafting": true,
    "use_persona": true
  },
  "lora": {
    "worldview": "lovecraft_worldview.txt"
  }
}
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Content lost | Increase `entailment_threshold` to 0.9, lower adapter scale |
| Style too weak | Increase adapter scale to 0.5-1.0 |
| Memorized output | Lower adapter scale, use earlier checkpoint |
| Out of memory (training) | Reduce `num_layers` to 16, enable `grad_checkpoint` |
| Out of memory (inference) | Use 7B model instead of 14B |

---

## License

MIT License
