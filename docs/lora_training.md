# LoRA Training Reference & Checklist

Complete guide for training a LoRA adapter to capture an author's writing style.

## Overview

This system uses **instruction back-translation** to train style: we teach the model to transform semantic descriptions ("what happens") into stylized prose ("how the author would write it"). The model learns to generate style, not just paraphrase.

**Key insight**: ~0.9M tokens (approximately 2 books) is sufficient for expert-level style emulation. More data doesn't improve quality and increases overfitting risk.

---

## Prerequisites Checklist

- [ ] Apple Silicon Mac (M1/M2/M3) - MLX only works on Apple Silicon
- [ ] Python 3.9+
- [ ] MLX installed: `pip install mlx mlx-lm`
- [ ] sentence-transformers (for corpus curation): `pip install sentence-transformers`
- [ ] Author's source text (plain .txt format, UTF-8)

---

## Step 1: Prepare Source Corpus

### 1.1 Obtain Author Text

Convert the author's works to plain text:
- ePub: Use Calibre to convert to .txt
- PDF: Use `pdftotext` or Calibre

**Critical: Clean the corpus manually before processing:**
- Remove prefaces, introductions, forewords (unless written by the author)
- Remove copyright notices, publisher information, ISBN pages
- Remove table of contents, chapter headers, page numbers
- Remove acknowledgments, about the author sections
- Remove any text not written in the author's characteristic style
- Keep only the author's actual prose (the "meat" of the writing)

### 1.2 Curate Corpus (Optional but Recommended)

For large corpuses (>2 books), curate to optimal size:

```bash
python scripts/curate_corpus.py \
    --input styles/sample_author_full.txt \
    --output styles/sample_author.txt \
    --target-tokens 900000
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-tokens` | 900000 | Target token count (~0.9M per research) |
| `--min-para-words` | 40 | Skip paragraphs shorter than this |
| `--no-cluster` | false | Skip semantic clustering, use sequential sampling |
| `--verbose` | false | Show detailed filtering statistics |

**What the script does:**
1. Filters out low-quality text (short paragraphs, OCR artifacts, fragments)
2. Uses embedding clustering to select diverse passages
3. Caps output at target token budget

**Quality filters applied:**
- Minimum 40 words per paragraph
- At least 2 complete sentences
- No excessive special characters (>10%)
- No excessive word repetition (>15%)
- No encoding artifacts

### Validation Checkpoint
- [ ] Corpus is clean plain text (no HTML, markdown, or formatting codes)
- [ ] Target size is ~600K-900K tokens (larger is not better)
- [ ] Text represents the author's characteristic style (not introductions, etc.)

---

## Step 2: Generate Training Data (Neutralization)

This is the critical step that creates instruction-response pairs for training.

```bash
python scripts/neutralize_corpus.py \
    --input styles/sample_author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name"
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-words` | 250 | Minimum words per training chunk |
| `--max-words` | 650 | Maximum words per training chunk |
| `--no-overlap` | false | Disable overlap between chunks |
| `--no-cleanup` | false | Skip LLM cleanup of chunk boundaries |
| `--llm` | deepseek | LLM provider: `deepseek` (best), `mlx`, or `ollama:model` |
| `--workers` | 1 | Parallel workers (DeepSeek/Ollama) |
| `--no-resume` | false | Start fresh, ignore checkpoint |

### LLM Provider Options

**DeepSeek (default, recommended - best instruction following):**
```bash
python scripts/neutralize_corpus.py \
    --input styles/sample_author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name" \
    --workers 8
```

**MLX (local, no API key needed):**
```bash
python scripts/neutralize_corpus.py \
    --input styles/sample_author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name" \
    --llm mlx
```

**Ollama (local with parallelization):**
```bash
python scripts/neutralize_corpus.py \
    --input styles/sample_author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name" \
    --llm ollama:qwen3:8b \
    --workers 4
```

### What This Script Does

1. **Segmentation**: Splits text into 250-650 word chunks with overlap
2. **Boundary cleanup**: Uses LLM to fix chunks cut mid-sentence
3. **Content description**: Generates semantic descriptions of each chunk:
   > "Describe in detail what is happening in this excerpt. Mention any characters and whether the voice is in first or third person. Maintain the order of sentences while describing."

4. **Creates training pairs**: Each pair contains:
   - `description`: Neutral content description (what happens)
   - `original`: Author's actual text (how they wrote it)
   - `word_count`: Length of original text

### Resuming Interrupted Processing

The script automatically saves checkpoints. If interrupted, just re-run the same command to resume:
```bash
# Resumes from checkpoint automatically
python scripts/neutralize_corpus.py \
    --input styles/sample_author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name"
```

To start fresh:
```bash
python scripts/neutralize_corpus.py \
    --input styles/sample_author.txt \
    --output data/neutralized/author.jsonl \
    --author "Author Name" \
    --no-resume
```

### Validation Checkpoint
- [ ] Output JSONL has reasonable number of examples (expect ~1500-3000 for 0.9M tokens)
- [ ] Descriptions are semantic (what happens) not paraphrases (different wording)
- [ ] Word counts are in 250-650 range
- [ ] Coverage ratio shown at completion is >90%

---

## Step 3: Train LoRA Adapter

Create a config.yaml file for training (see `data/training/lovecraft/config.yaml` for a template):

```yaml
# data/training/author/config.yaml
model: "mlx-community/Qwen3-8B-Base-bf16"
train: true
data: "data/training/author"

batch_size: 1
grad_accumulation: 2
iters: 3000              # ~0.87 epochs
learning_rate: 1e-5
num_layers: -1

lora_parameters:
  rank: 64
  scale: 2.0
  dropout: 0.15
  keys:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"
    - "mlp.gate_proj"
    - "mlp.up_proj"
    - "mlp.down_proj"

adapter_path: "lora_adapters/author"
save_every: 500
steps_per_report: 50
steps_per_eval: 200
seed: 42
```

Then run training:
```bash
mlx_lm.lora --config data/training/author/config.yaml
```

**Key Parameters:**
| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `iters` | ~0.87 epochs | Prevents overfitting |
| `batch_size` | 1 | Learns individual examples better |
| `learning_rate` | 1e-5 | Lower with high rank for stability |
| `rank` | 64 | Higher = more capacity for complex syntax |
| `dropout` | 0.15 | Prevents keyword memorization |

### Training Format

The training data uses this format:
```
Write a {word_count} word excerpt about the content below emulating the style and voice of {author}

{content_description}

{original_text}
```

This teaches the model: given a description and target length, generate text in the author's style.

### Validation Checkpoint
- [ ] Training completes without errors
- [ ] Validation loss decreases during training
- [ ] `adapters.safetensors` file created in output directory

---

## Step 4: Test the Adapter

Use the style transfer pipeline to test:
```bash
python restyle.py test_input.md -o test_output.md \
    --adapter lora_adapters/author \
    --author "Author Name" \
    --verbose
```

### Validation Checkpoint
- [ ] Generated text sounds like the author
- [ ] Content from the prompt is preserved
- [ ] No obvious repetition or degeneration
- [ ] Grammar and coherence are good

---

## Complete End-to-End Example

Train a Carl Sagan adapter:

```bash
# Step 1: Curate corpus
python scripts/curate_corpus.py \
    --input data/corpus/sagan_full.txt \
    --output data/corpus/sagan.txt \
    --target-tokens 900000

# Step 2: Index in ChromaDB
python scripts/load_corpus.py \
    --input data/corpus/sagan.txt \
    --author "Carl Sagan"

# Step 3: Generate training data
python scripts/generate_flat_training.py \
    --corpus data/corpus/sagan.txt \
    --author "Carl Sagan" \
    --output data/training/sagan

# Step 4: Train LoRA (create config.yaml first, see template above)
mlx_lm.lora --config data/training/sagan/config.yaml

# Step 5: Test with style transfer
python restyle.py input.md -o output.md \
    --adapter lora_adapters/sagan \
    --author "Carl Sagan"
```

---

## Hyperparameter Reference

### Training Hyperparameters (from Research)

| Setting | Value | Reasoning |
|---------|-------|-----------|
| Batch Size | 1 | Prevents averaging gradients across diverse examples |
| Grad Accumulation | 2 | Effective batch size of 2 |
| Learning Rate | 1e-5 | Lower with high rank for stability |
| Iterations | ~0.87 epochs | Prevents overfitting |
| LoRA Rank | 64 | Higher capacity for complex syntax |
| LoRA Scale | 2.0 | Strong style signal |
| Dropout | 0.15 | Forces structural learning over memorization |

### Chunk Size Parameters

| Setting | Value | Reasoning |
|---------|-------|-----------|
| Min chunk words | 250 | Captures paragraph-level style |
| Max chunk words | 650 | Fits in context window with overhead |
| Overlap | Yes | Style lives in transitions between paragraphs |

### Inference Parameters

| Setting | Value | Reasoning |
|---------|-------|-----------|
| Temperature | 1.0 | Allows creative vocabulary selection |
| Top-p | 0.9 | Standard nucleus sampling |
| Repetition penalty | 1.1 | Prevents degeneration |

---

## Troubleshooting

### "MLX not available"
- MLX only works on Apple Silicon Macs
- Install with: `pip install mlx mlx-lm`

### Training loss not decreasing
- Check corpus quality (remove non-representative text)
- Ensure descriptions are semantic, not paraphrases
- Try increasing learning rate slightly (1.5e-4)

### Generated text doesn't match style
- Need more diverse training examples
- Corpus may be too short (target 0.9M tokens)
- Try more epochs (up to 3) for smaller corpuses

### Generated text repeats or degenerates
- Reduce temperature to 0.7
- Increase repetition penalty to 1.2
- May be overfitting - reduce epochs or use more data

### Output too long/short
- Model learns length from word_count in training
- Ensure training data has accurate word counts
- At inference, specify target word count in prompt

### Memory issues during training
- Reduce --max-seq-length to 1024
- Use a smaller base model (4-bit quantized)
- Close other applications

---

## File Structure After Training

```
lora_adapters/
└── author/
    ├── adapters.safetensors    # LoRA weights
    ├── metadata.json           # Training metadata
    └── data/
        ├── train.jsonl         # Training examples
        └── valid.jsonl         # Validation examples

data/neutralized/
└── author.jsonl                # Content descriptions
```

---

## Resources

- Paper: [Style Transfer with Instruction-Tuned LLMs](https://arxiv.org/pdf/2510.13939)
- Example dataset: [Gertrude Stein Style SFT](https://huggingface.co/datasets/MuratcanKoylan/gertrude-stein-style-sft)
- Training details: [Project Blog](https://muratcankoylan.com/projects/gertrude-stein-style-training)
- Pipeline code: [GitHub](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering/tree/main/examples/book-sft-pipeline)
