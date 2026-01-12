# LoRA Training Reference & Checklist

Complete guide for training style transfer LoRA adapters using Qwen 2.5.

## Critical: Training/Inference Distribution Matching

**The most important insight**: Output quality depends on matching inference conditions to training conditions exactly.

See `docs/lora_training_strategy.md` for the complete analysis. Key requirements:

| Must Match | Training | Inference |
|------------|----------|-----------|
| **Input perspective** | First-person narrative ("I saw", "I found") | `narrativize_input: true` |
| Input perturbation | 8% noise | `apply_input_perturbation: true` |
| Persona frames | `PERSONA_FRAMES` dict | `prompts/{author}_worldview.txt` |
| Content classifier | `classify_content_type()` | `src/utils/content_classifier.py` |
| Scale | 2.0 | 2.0 in config.json |

**CRITICAL Pipeline Order**: Narrativize must happen BEFORE RTT, not after:

```
Correct:  Input → Narrativize → RTT → Perturb → LoRA  ✓
Wrong:    Input → RTT → Narrativize → Perturb → LoRA  ✗
```

The LoRA was trained on RTT output. If narrativize happens after RTT, it destroys the neutral format the LoRA expects. The narrativize step converts impersonal input to first-person BEFORE RTT neutralizes it.

## Overview

This system uses **Persona Simulation** rather than simple Style Transfer. Instead of asking the model to "Rewrite X," we place it inside a narrative frame (e.g., "You are an engineer analyzing a system failure") to force it to adopt the target worldview.

**Key Strategy**:

* **Model**: Qwen 2.5 14B Base (4-bit).
* **Technique**: Triad Strategy (original + snowflake + robustness).
* **Objective**: Force the model to creatively reconstruct degraded input into rich prose.

---

## Prerequisites Checklist

* [ ] Apple Silicon Mac (M1/M2/M3) - 16GB+ RAM (32GB+ Recommended)
* [ ] Python 3.9+
* [ ] MLX installed: `pip install mlx mlx-lm`
* [ ] Author's source text (plain .txt format, UTF-8)

---

## Step 1: Prepare Source Corpus

### 1.1 Obtain & Clean Text

Convert the author's works to plain text.
**Critical:** Remove anything that breaks the "Fourth Wall."

* Remove: Prefaces, Copyrights, Chapter Titles, Footnotes.
* Keep: Only the raw prose (narrative and exposition).

### 1.2 Curate Corpus (The "Chunking" Strategy)

We need chunks that are long enough to capture *paragraph structure* (topic sentence  evidence  cosmic conclusion).

```bash
python scripts/curate_corpus.py \
    --input styles/unconducted_chorus_full.txt \
    --output styles/unconducted_chorus.txt \
    --target-tokens 900000 \
    --min-para-words 60 

```

*Note: We increased `min-para-words` to 60 because short paragraphs don't carry enough style signal.*

---

## Step 2: Generate Training Data (Persona Injection)

This step creates the "Many-to-One" training pairs with perspective variations.

```bash
python scripts/generate_flat_training.py \
    --corpus styles/unconducted_chorus.txt \
    --author "The Unconducted Chorus" \
    --output data/training/lovecraft

# Skip perspective variations (faster, fewer examples)
python scripts/generate_flat_training.py \
    --corpus styles/unconducted_chorus.txt \
    --author "The Unconducted Chorus" \
    --output data/training/lovecraft \
    --skip-perspective
```

### The Enhanced Triad Strategy

The script generates multiple variation types per paragraph:

**Content Variations:**
1. **Original (Anchor):** Real author text in first-person singular
2. **Snowflake:** Topic swap to mundane activity (preserves structure, different content)
3. **Robustness:** Same text with heavy input perturbation

**Perspective Variations (NEW):**
4. **First Person Plural:** "we saw" instead of "I saw"
5. **Third Person:** "the observer saw" instead of "I saw"
6. **Impersonal:** "it was observed" instead of "I saw"

Perspective variations teach the LoRA that style is independent of POV. This allows inference to work with ANY input perspective, eliminating the need for the `narrativize_input` step.

### Input Variants per Anchor

For each anchor, we generate multiple input variants:
1. **Standard Neutral:** RTT-neutralized text (baseline)
2. **Info-Dropout:** No adjectives (model must invent them)
3. **Abstract:** Concrete nouns replaced with [THING] placeholders

**Validation Checkpoint:**

* [ ] Output `train.jsonl` has sufficient examples (varies by corpus size)
* [ ] Inputs are "neutral" or "broken"; Outputs are "lush" and "styled"
* [ ] Check `valid.jsonl` to ensure no "I am an AI" refusals exist in the target text
* [ ] Perspective variations cover all three types (plural, third, impersonal)

---

## Step 3: Train LoRA Adapter

We use the **Qwen 2.5 14B Base (4-bit)** model. This is the "Goldilocks" model: smarter than 8B, faster than 32B, and fits on consumer hardware with high-rank adapters.

### 3.1 Convert the Base Model

You must convert the base model manually, as it is not always available pre-quantized.

```bash
mlx_lm.convert --hf-path Qwen/Qwen2.5-14B --q-bits 4 --mlx-path ./models/Qwen2.5-14B-Base-4bit-MLX

```

### 3.2 Configuration (`config.yaml`)

Create `data/training/lovecraft/config.yaml`. This is tuned for **High Fidelity** (Rank 64) on 16GB-32GB RAM machines.

```yaml
# data/training/lovecraft/config.yaml

# 1. Model & Data
model: "./models/Qwen2.5-14B-Base-4bit-MLX"
train: true
data: "data/training/lovecraft"
fine_tune_type: lora

# 2. Hyperparameters
# Batch Size 1 allows the model to learn from specific, dense examples.
# Grad Accumulation 4 stabilizes the updates (Effective Batch = 4).
batch_size: 1
grad_accumulation: 4

# Duration:
# 3,770 examples / 4 (Eff Batch) = 942 steps per epoch.
# Target: ~2.2 Epochs (2,100 steps) to burn in the persona.
iters: 2100

# Optimization
learning_rate: 1e-5
grad_checkpoint: true    # CRITICAL: Enables training on limited RAM

# 3. LoRA Architecture (High Fidelity)
lora_parameters:
  rank: 64               # High capacity for complex sentence structures
  scale: 2.0             # Strong style signal
  dropout: 0.1           # Prevents verbatim memorization
  
  # Target ALL Linear Layers (Attn + MLP)
  # Learns Rhythm (Attn) AND Vocabulary (MLP)
  keys:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"
    - "mlp.gate_proj"
    - "mlp.up_proj"
    - "mlp.down_proj"

# 4. Checkpointing
adapter_path: "lora_adapters/lovecraft"
save_every: 200
steps_per_report: 10
steps_per_eval: 200
seed: 42

```

### 3.3 Run Training

```bash
mlx_lm.lora --config data/training/lovecraft/config.yaml

```

**Memory Management:**

* If you get an **OOM (Out of Memory)** error:
1. First, ensure `grad_checkpoint: true` is set.
2. Second, reduce `rank` to **32**.
3. Third, reduce `rank` to **16** (but keep `scale: 2.0`).



---

## Step 4: Inference (The Persona Prompt)

A "Persona Adapter" requires a "Persona Prompt" to activate. Do not just say "Rewrite this."

**Usage Pattern:**

```python
prompt = """
[SYSTEM ROLE]:
You are the Systems Theologian. You view the world as a machine governed by indifferent geometry.
You do not see "bad luck"; you see "entropy."
You are analyzing the following input.

[INSTRUCTION]:
Explain the input concept.
- Use mechanical metaphors (gears, rot, friction).
- Do not be cheerful. Be precise and grave.
- Conclude by linking the concept to the inevitable heat death of the universe.

[INPUT]:
{user_input}
"""

# Run with MLX
subprocess.run(f"mlx_lm.generate --model ./models/Qwen2.5-14B-Base-4bit-MLX --adapter lora_adapters/lovecraft --prompt '{prompt}'")

```

---

## Troubleshooting Reference

| Issue | Diagnosis | Fix |
| --- | --- | --- |
| **"Command buffer execution failed: Insufficient Memory"** | You ran out of VRAM. | 1. Enable `grad_checkpoint: true` 2. Reduce Rank to 16. 3. Use `max_seq_length: 1024`. |
| **Model outputs gibberish / loops** | "Fried" weights (Overfitting). | Reduce `iters` (you trained too long). Load an earlier checkpoint. |
| **Model refuses to generate ("I cannot...")** | Safety Refusal. | You are using an Instruct model. **Switch to Base model.** |
| **Style is too weak** | Scale mismatch. | Set `scale: 2.0` to match training. Check `use_persona: true`. |
| **Output is mechanical/formulaic** | Distribution mismatch. | Enable `apply_input_perturbation: true`. Verify persona frames match training. |
| **Output is passive/descriptive/impersonal** | Input perspective mismatch. Training used first-person narrative ("I saw"), inference uses impersonal ("We trace"). | Enable `narrativize_input: true`. This converts "We trace..." → "I traced..." before LoRA. |
| **Output doesn't expand** | LoRA can't expand. | Enable `expand_for_texture: true`. LoRA trained on similar-length pairs. |
| **Wrong persona frame type** | Content misclassified. | Both must use `src/utils/content_classifier.py`. Check spaCy NER output. |

---

## Final Quality Check

* [ ] **Rhythm Test:** Does the output sound "breathless" or "complex"? (Check sentence length variance).
* [ ] **Vocab Test:** does it use specific author words (e.g., "geometry," "indifferent," "mechanism")?
* [ ] **Hallucination Test:** If you give it a bare noun ("The Stock Market"), does it invent a metaphor ("A shifting ocean of greed")? If yes, it works.