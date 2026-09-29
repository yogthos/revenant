# LoRA Training Findings: Why Convergent Training Failed

This document captures key findings from our LoRA training experiments and documents the correct approach.

## Executive Summary

**Convergent training (multiple paraphrases → same output) does not work for style transfer.**

The model learns to copy input with minor word substitutions rather than learning style transformation. The correct approach is **instruction back-translation**: semantic descriptions → styled prose.

---

## Failed Approach: Convergent Training

### What We Tried

Used `generate_convergent_training_llm.py` to create training data where multiple neutral paraphrases map to the same styled output:

```
input_variation_1 → author_paragraph_A
input_variation_2 → author_paragraph_A  (same output)
input_variation_3 → author_paragraph_A  (same output)
```

### Training Data Example (Actual)

```json
{
  "messages": [
    {"role": "system", "content": "Rephrase in mixed authors's prose style."},
    {"role": "user", "content": "One day, might intelligent computers wonder about their own beginnings? Could one discover the heretical idea that they originated from an earlier, organic carbon-based life form..."},
    {"role": "assistant", "content": "Could it be that one far-off day intelligent computers will speculate about their own lost origins? Will one of them tumble to the heretical truth, that they have sprung from a remote, earlier form of life..."}
  ]
}
```

### Why It Failed

1. **Input ≈ Output**: The input is already styled prose, just slightly reworded
2. **92-96% word overlap**: Model learns that input and output are nearly identical
3. **Learned behavior**: Copy input with minor substitutions ("might" → "could it be")
4. **Post-copy garbage**: Model doesn't know when to stop, generates garbage tokens

### Observed Output

```
Input:  "The IMF warns that systemic risks are building in shadow banking."
Output: "The IMF warns that systemic risks are building in shadow banking.ประทับอาศัย
        The IMF warns that systemic risks are building in shadow banking.เพื่อยังคง..."
```

The model copies, generates garbage, then repeats in a loop.

---

## Correct Approach: Instruction Back-Translation

### The Key Insight

From the Gertrude Stein style training research:
> "Instruct-tuning creates response patterns that resist style overwriting. Base models are blank canvases."

And critically:
> "Style lives in the transitions between passages."

### How It Works

Instead of paraphrase → prose, use **description → prose**:

```
INPUT:  "Someone contemplates whether future artificial intelligences might
         wonder about their origins. A hypothetical scenario exploring the
         possibility that silicon-based life descended from organic precursors."

OUTPUT: "Could it be that one far-off day intelligent computers will speculate
         about their own lost origins? Will one of them tumble to the heretical
         truth, that they have sprung from a remote, earlier form of life..."
```

### Why It Works

| Aspect | Convergent (Failed) | Back-Translation (Works) |
|--------|---------------------|--------------------------|
| Input type | Paraphrase (prose) | Description (meta) |
| Structural similarity | ~95% overlap | ~10% overlap |
| What model learns | Copy with substitution | Generate from description |
| Style signal | Noise (2% of tokens) | The entire output (100%) |

### The Description Prompt

From `generate_flat_training.py`:

```
Summarize what happens in 50-100 words. You MUST replace ALL proper nouns
with generic descriptions.

ABSOLUTE RULES:
1. NO CHARACTER NAMES (John, Mary) → "a young man", "the woman"
2. NO PLACE NAMES (London, Arkham) → "a city", "a dark forest"
3. Start directly with action, never with "The passage..."
```

This creates descriptions that are **structurally different** from the prose, forcing the model to learn style generation.

---

## Additional Findings

### Mixed Authors Don't Work

Training with `"mixed authors"` as the author name dilutes the style signal:
- Model can't learn any specific voice
- Conflicting patterns from different authors cancel out
- Result: bland output with no distinctive style

**Solution**: Train per-author LoRAs, or use clear author labels in prompts.

### Base Model Required

From the research:
- Instruction-tuned models have ingrained response patterns
- These patterns resist style overwriting
- Base models (e.g., `Qwen3-8B-Base`) are "blank canvases"

**We correctly used a base model**, but combined it with the wrong training format.

### Optimal Hyperparameters (From Research)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 1 | Prevents averaging across diverse examples |
| Learning rate | 1e-4 to 5e-4 | Aggressive style imprinting |
| Epochs | 1-3 | 1 for large corpus, 3 for small |
| LoRA rank | 32 | Good capacity for style patterns |
| Chunk size | 150-400 words | Style lives in transitions |

### Temperature at Inference

Our experiments found:
- `temp=0.2`: Model loops/repeats before completing sentences
- `temp=0.4`: Better sentence completion, less looping
- `temp=0.5+`: More hallucination risk

---

## Correct Training Pipeline

### Step 1: Prepare Corpus

Clean author text (see `docs/input_text_format.md`):
- Remove headers, page numbers, non-prose
- Keep only characteristic prose
- Target ~900K tokens (2 books worth)

### Step 2: Generate Descriptions (Neutralization)

```bash
python scripts/generate_flat_training.py \
    --corpus data/corpus/author.txt \
    --author "Author Name" \
    --output data/training/author
```

This creates neutral → styled pairs via RTT neutralization. By default, it also generates:
- Topic variations (snowflake) to isolate style from content
- Robustness variations with input perturbation
- Perspective variations (first_person_plural, third_person, impersonal)

Use `--skip-variation` and `--skip-perspective` flags to disable these.

### Step 3: Train LoRA

Create a `config.yaml` for training (see `data/training/lovecraft/config.yaml` for template), then:

```bash
mlx_lm.lora --config data/training/author/config.yaml
```

### Step 4: Test

```bash
python restyle.py test_input.txt -o test_output.txt \
    --adapter lora_adapters/author \
    --author "Author Name"
```

---

## Scripts to Use vs. Avoid

### Use These

| Script | Purpose |
|--------|---------|
| `generate_flat_training.py` | Generate training data via RTT neutralization |
| `curate_corpus.py` | Filter and size corpus optimally |
| `load_corpus.py` | Index corpus in ChromaDB |

### Avoid These (Deprecated)

| Script | Why Deprecated |
|--------|----------------|
| `generate_convergent_training_llm.py` | Creates paraphrases, not descriptions |
| `generate_convergent_training.py` | Rule-based version, same problem |
| `prepare_lora_training.py` | For convergent format, not needed |
| `generate_repair_training.py` | Repair approach didn't help |

---

## Key Takeaways

1. **Paraphrase ≠ Description**: Paraphrases are too similar to output; descriptions force style generation
2. **One author at a time**: Mixed training dilutes signal; train per-author
3. **Base models only**: Instruction-tuned models resist style overwriting
4. **Small chunks with overlap**: Style lives in transitions (150-400 words)
5. **Description must be structurally different**: Replace names, summarize action, not just reword

---

## Retold Inputs Teach Vocabulary, Not Structure

The first Hemmingway-1 Russell run trained on RTT retellings of Russell's
paragraphs. Retellings keep the original's sentence order and argument, so
the model only had to restore Russell's vocabulary and rhythm on a structure
it was handed. Given LLM-written text it kept the LLM structure, and GPTZero
still flagged the output. Adapter scale (1.0-2.0) and temperature changed
word choice, not structure.

The fix is structural distance between input and target, the same lesson as
description -> prose above: llm_style rows start from an LLM-register
rewrite of the author's paragraph (see style_transfer_training.md).

## Fact Hallucination Problem

### The Issue

Even with correct training format, LoRA-adapted models hallucinate facts:
- Numbers get converted: `42 years old` → `twenty-four`
- Times get spelled out: `3:47 PM` → `three forty-seven`
- Dates get changed: `June 22, 1949` → random other dates
- Names get substituted: `Dr. Warren` → different names
- Completely new "facts" get invented

This happens at **all LoRA scales**, even 0.3. The base model also exhibits this behavior.

### Why It Happens

1. **Training data has no constraints on facts** - The LoRA learns style patterns but not fact preservation
2. **The model treats numbers as style elements** - Stylistic authors often use words instead of digits
3. **No grounding mechanism** - The model can't verify facts against source

### Recommendations

For fact-heavy text:
- Use lower LoRA scale (0.5-1.0) to reduce hallucination severity
- Higher LoRA scales (>1.5) risk memorization of training passages

---

## References

- [Gertrude Stein Style Training](https://muratcankoylan.com/projects/gertrude-stein-style-training)
- [Style Transfer Paper](https://arxiv.org/pdf/2510.13939)
- [Example Dataset](https://huggingface.co/datasets/MuratcanKoylan/gertrude-stein-style-sft)
