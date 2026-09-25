# Style Transfer LoRA Training Guide

This document describes how to train a LoRA adapter for literary style transfer.
The approach is model-agnostic — it works with any base model (Qwen 2.5, 3.5, etc.)
as long as the training data and inference pipeline match.

## Core Principles

### 1. Training Data Format

The model learns from (neutralized_input → styled_output) pairs:

```
System: {persona_frame} Write approximately {N} words. {constraints}
User:   {RTT-neutralized text, ~N words}
Assistant: {Original author text, ~N words}
```

This teaches:
- **Length preservation**: input ≈ output words (~1.21x average expansion)
- **Style application**: vocabulary, rhythm, sentence structure
- **No content expansion**: the model never learns to add new ideas

**Wrong approach** (causes hallucination):
```
User: "A passage about X, Y, Z" (15 words)
Assistant: [Full 300 word passage]
```

### 2. Distribution Matching

The LoRA's quality depends on matching inference conditions to training conditions:

| Factor | Must Match? | Notes |
|--------|-------------|-------|
| Persona frames | YES | Inference worldview file must use exact training frames |
| Content classifier | YES | Narrative vs conceptual detection |
| Input perturbation (8%) | YES | `apply_input_perturbation: true` at inference |
| RTT neutralization | YES | Same neutralizer at training and inference |
| Perspective conversion | YES | Happens BEFORE RTT, not after |
| Word count ratio | Fixed | ~1.21x, cannot change via config |
| LoRA scale | Baked in | config.json `scale` multiplies it; 1.0 = as trained |

### 3. Input Perturbation Is Critical

Training data uses 8% input perturbation (typos, article drops, synonym swaps). Noise never drops words that carry meaning.
Without perturbation at inference, the model receives clean text and produces mechanical output
because it has "no room" to exercise creative reconstruction.

### 4. Pipeline Order

```
Input → Perspective Conversion → RTT Neutralization → Perturb → LoRA
```

**Wrong** (destroys RTT output):
```
Input → RTT → Perspective Conversion → Perturb → LoRA ❌
```

## Training Data Generation

### Triad Strategy

For each original paragraph, generate 3+ variations:

1. **Anchor**: Original author text → teaches vocabulary and natural expression
2. **Snowflake**: Topic swap (author structure applied to different subject) → teaches that style applies to any content
3. **Robustness**: Heavy input perturbation (15%) → prevents overfitting to specific input words

### Snowflake Topics

Snowflake topics should match the content the LoRA will encounter at inference:
- For **narrative** authors (Lovecraft): mundane activities, atmospheric descriptions
- For **expository** authors (Russell): philosophical arguments, scientific concepts, the book's actual themes

Use `--snowflake-topics` to load a custom topic list per author:
```bash
python scripts/generate_flat_training.py \
    --snowflake-topics data/training/russell/snowflake_topics.py
```

### Perspective Variations

Generate the same content in different perspectives to teach style independence from POV:
- `first_person_plural`: "we saw" instead of "I saw"
- `third_person`: "the observer saw"
- `impersonal`: "it was observed"

### Overlapping Chunks

Stylistic markers concentrate at sentence boundaries. Overlapping chunks expose the model
to more beginning/ending patterns where style lives:
```
Config: min_words=150, max_words=400, overlap_sentences=2
```

Only applied to `original` type entries (continuous narrative). Snowflakes and perspective
variants are kept separate to avoid Frankenstein text.

### Quality Filtering and Split

`generate_flat_training.py` runs this at the end for LlamaFactory output. To rerun it:

```bash
python scripts/filter_training_data.py data/training/author/train.jsonl
```

It drops rows whose input kept an entity placeholder or stray leading punctuation,
rows with too much lexical bleed, rows with an output/input word ratio over 2.0 or
inputs under 15 words, and (unless `--no-nli`) rows that fail a two-way,
sentence-level entailment check: the target must not state things the input lacks,
and the input must not state things the target lacks.

It then writes `LlamaFactory/train.jsonl`, `LlamaFactory/val.jsonl` and
`dataset_info.json`. Validation holds out whole source paragraphs, so overlapping
chunks and many-to-one variants of a held-out paragraph never appear in train.
Point the yaml at it with `eval_dataset: <name>_val` rather than `val_size`.

### Persona Frames

Frames are split into NARRATIVE and CONCEPTUAL to avoid instruction-content mismatch.
A narrative about characters should not use "Explain the concept..." prompts.

Frames must be:
1. Defined in `PERSONA_FRAMES` dict in `generate_flat_training.py` for training
2. Copied exactly to `prompts/{author}_worldview.txt` for inference

### Anti-AI Constraints

Every training entry includes tiered constraints to prevent LLM-speak:
- **Always** (100%): Ban "Moreover", "Furthermore", "Therefore", etc.
- **Frequent** (70%): No topic sentences, no numbered lists
- **Rotating** (40%): One random stylistic constraint (fragments, rhetorical questions, etc.)

### Instruction Template Diversity

Using identical instruction prompts across training examples causes **attention collapse**:
the model learns to respond to the prompt pattern rather than learning style. The
[Stein study](https://muratcankoylan.com/projects/gertrude-stein-style-training/)
used 15 templates × 5 system prompts (75 combinations) to prevent this.

Our pipeline achieves diversity through:
- Multiple persona frames (3+ narrative, 3+ conceptual per author)
- Random constraint selection (ALWAYS + 70% FREQUENT + 40% ROTATING)
- Optional rhetorical skeleton (50% chance)
- Random word count targets

Combined, this produces high instruction variety. For blended authors with fewer frames,
consider adding more frame variations to maintain diversity.

## What We Learned (Experimental Findings)

### Convergent Training Failed

Training with (paraphrase → same_output) pairs where multiple paraphrases map to one
output caused the model to learn copy-paste with minor word substitutions (92-96% overlap).
The model couldn't distinguish "make this sound like the author" from "copy the input."

### Back-Translation Works

The current approach — RTT neutralization that strips all style, then training the model
to reconstruct the style — works because the structural overlap between input and output
is only ~10%, forcing the model to learn actual style patterns.

### Mixed Authors Don't Work for Training Output

Training with non-target-author text as OUTPUT (e.g., Sagan paragraphs as output when
training a Lovecraft LoRA) teaches the wrong style. The output must always be in the
target author's voice. Use snowflake variations instead — target author's structure
applied to diverse topics.

### Blended Author Styles

For synthetic authors that blend two real authors (e.g., "Howard Russell" = Russell
substance + Lovecraft mood), training data is created by **sentence-level corpus stitching**:

1. **Sentence splicing**: Embed both corpora at sentence level, find semantically
   compatible sentences via cosine similarity, stitch into chimera paragraphs
2. **Vocabulary transplant**: Replace evaluative adjectives with curated atmospheric
   vocabulary from the mood author (e.g., "complete" → "cavernous", "strange" → "eldritch")
3. **Subject alignment**: Minimal LLM surgery to align pronouns/referents across stitched sentences
4. **Blended persona frames**: Combine the urgency/atmosphere of one author's frames with
   the analytical structure of the other's

Technical terms, classificatory adjectives, and Latin phrases must be protected from
vocabulary transplant via blocklist. The vocabulary pool should be curated for atmospheric
effect, not auto-extracted by frequency (frequency-distinctive words are often non-atmospheric).

See `scripts/experiment_blend_sentences.py` for the blending pipeline.

### Paragraph Length Must Match Inference

If training data is predominantly 200-400 word paragraphs but inference processes 60-100
word paragraphs (common in chapter-length works), the LoRA will underperform on short inputs.
Include training examples across the full range the model will encounter, with emphasis
on the 60-150 word sweet spot (per `docs/input_text_format.md`).

### Overlapping Chunks Improve Quality

Moving from isolated paragraphs to overlapping chunks (150-400 words, 2-sentence overlap)
doubled training examples from identical source material and significantly improved style
quality in the [Stein study](https://muratcankoylan.com/projects/gertrude-stein-style-training/).
The model learns better from "more edges" — more sentence/paragraph boundaries where
stylistic patterns are most distinctive.

### Prompting Cannot Replace Fine-Tuning

The Stein study found that prompting frontier models (Claude, GPT-4) for 400-word style
passages achieved only 5% human perception (95% AI detection), matching research showing
97% detection despite expert prompts. Fine-tuning achieved 70% human perception. Prompting
can approximate style superficially but cannot capture the deep structural patterns
(sentence rhythm, clause nesting, punctuation habits) that make prose feel authentically
authored.

### Base Models Required

Instruct-tuned models resist style overwriting because they have built-in style biases
(assistant-speak, hedging, balanced arguments). Base models provide a clean slate.
The Stein study independently confirmed this: "instruct-tuning creates response patterns
that resist style overwriting."

### Fact Hallucination

The model corrupts numbers, dates, and proper names at all LoRA scales. This is handled
at inference by the semantic verification pipeline, not during training.

### Validation: Modern Scenario Testing

To verify the model learned style rather than memorizing content, generate text for
scenarios impossible in the author's time period. If the model produces stylistically
authentic prose about modern topics (AI, social media, space travel), it learned the
style patterns, not the subject matter. Snowflake topic variations test this during
training; modern-topic inference tests it in production.

## Hyperparameter Guide

### Rank — Rank 256 is Critical for Literary Style

**For literary style transfer, rank 256 is the empirically-validated sweet spot.**
Lower ranks only capture surface features; higher ranks over-apply style into
incoherence. The gap between rank 128 and rank 256 is described as a "huge jump"
in actual style capture, not a marginal improvement.

| Rank | Style Quality | Use Case |
|------|--------------|----------|
| 16 | Minimal — picks up punctuation quirks only | Classification, simple format compliance |
| 64 | Basic — captures vocabulary and basic rhythm | Simple style transfer (Lovecraft adapter proof point) |
| 128 | Mild — "becoming a bit more verbose" | Insufficient for serious literary style |
| **256** | **Optimal — "huge jump" in style capture, best coherence/spirit balance** | **Default choice for literary style transfer** |
| 512 | Incoherent — over-application, "confusingly laid out and nonsensical" | Avoid |

**Why rank 256 is non-negotiable for literary style:**

From the [RunPod Nabokov study](https://www.runpod.io/blog/effects-of-rank-epochs-learning-rate-textual-loras)
(Llama-2-7B, ~800 pages of Nabokov, tested r=16/128/256/512):

> "At rank 256, we see a huge jump in Nabakov-ness with the parentheticals,
> hyphens, and character descriptiveness... provides the best balance between
> coherence and spirit."

> "Changing the rank is the largest adjustment you can make on how strong a
> LoRA is at altering the text."

Lower ranks captured "only stylistic cues" while rank 256 captured "facts and
vocabulary" alongside style. At rank 128 the study reported only mild verbosity
increase — not genuine style capture.

Independent confirmation from [Sebastian Raschka's practical tests](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
on instruction tuning (tested r=8/32/64/128/256/512):

> "r=256 with alpha=512 produced optimal performance, contrary to conventional
> wisdom."

> "The more diverse the tasks in the dataset, the larger the r should be."
> — implication: blended styles (multi-author) especially need higher rank

**Data requirements for rank 256:**

Data-per-rank ratio benchmarks (examples per rank unit):
- Stein study (minimum viable): 18 ex/rank → produced 70% human perception
- Lovecraft adapter (production): 59 ex/rank → production quality
- Our Howard Russell target: 18-36 ex/rank depending on dataset size

At rank 256 with ~4,600 examples (18 ex/rank), mitigate overfitting risk with:
- bf16 LoRA (no quantization noise)
- NEFTune alpha 5.0 (embedding noise)
- lora_dropout 0.1
- Low learning rate (1e-5 with alpha/rank=2.0 gives effective 2e-5)
- Eval every 200 steps with early stopping on val loss rise

**Do not compromise on rank 256 for style transfer to save memory.** If VRAM
is tight, first try: gradient checkpointing, 8-bit optimizer, cutoff_len 1536,
or disabling packing. Only fall back to rank 128 as a last resort — it will
produce noticeably weaker style capture.

**Alpha ratio:** Standard 2:1 (alpha=512 at rank=256) works. Raschka's tests
found 0.5:1 ratio also effective for some tasks. With rsLoRA, use alpha=sqrt(rank)
instead (see rsLoRA section below).

### rsLoRA (Rank-Stabilized LoRA)

**Required for rank ≥ 64.** Standard LoRA scales by `alpha/rank`, which kills gradient
flow at high ranks. rsLoRA scales by `alpha/sqrt(rank)`.

**Critical alpha setting with rsLoRA:**
- Standard LoRA: `alpha = rank` gives scaling = 1.0 (neutral)
- rsLoRA: `alpha = rank` gives scaling = `sqrt(rank)` (TOO HIGH)
- rsLoRA: `alpha = sqrt(rank)` gives scaling = 1.0 (correct)

| Rank | Correct alpha (rsLoRA) | Effective scaling |
|------|----------------------|-------------------|
| 64 | 8 | 1.0 |
| 128 | ~11 | 1.0 |
| 256 | 16 | 1.0 |

**Failure mode:** `alpha=256, rank=256, rsLoRA=true` gives scaling = 16x.
This caused gradient explosion (loss 132,000 at step 10, then 0.0 forever).

### Learning Rate

With rsLoRA at neutral scaling (1.0x):
- **2e-5**: Safe starting point for 30B+ models
- **5e-5**: Aggressive but may work with gradient clipping

Without rsLoRA:
- **1e-4**: Standard for LoRA SFT

### NEFTune

`neftune_noise_alpha: 5.0` — adds random noise to embeddings during training.
Prevents memorization of exact phrases, forces learning generalized style patterns.

### Gradient Clipping

`max_grad_norm: 0.3` — essential for MoE models. Router networks produce outlier
gradients in early training that can corrupt weights without clipping.

### Batch Size

Smaller effective batch = spikier gradients = better style quirk capture.
Target effective batch size of 4-8.

With DDP (multi-GPU): effective_batch = num_gpus × per_device_batch × grad_accum.

### Epochs

- 2 epochs: minimum (may underfit)
- 3 epochs: recommended starting point
- Watch eval loss — stop if it rises for 2+ eval intervals

### Dropout

`lora_dropout: 0.1` — higher than default 0.05 to prevent overfitting with high rank.

### Temperature at Inference

- 0.2: loops and repetition
- 0.4: better coherence
- 0.5+: hallucination risk
- 0.6-0.8: recommended for style transfer (config-dependent)

## Future Investigation

Ideas from [muratcankoylan/book-training](https://github.com/muratcankoylan/book-training)
that could improve training quality. Not yet implemented.

### Scene Description as Input Variant

Their pipeline uses an LLM to generate a 2-3 sentence scene description of each chunk,
then uses that as the user prompt ("Write in {author}'s style: {scene}"). We use
RTT-neutralized text as input. Adding a scene-description variant alongside our existing
many-to-one variants (standard, info_dropout, abstract) would teach the model to expand
from brief summaries — not just restyle word-for-word neutral text. This could improve
handling of short input paragraphs where the neutral text is thin.

### Template-Level Variation per Chunk — Already Implemented

They generate 2 variants per chunk using different prompt/system template combinations
mapping to the same output text. Our pipeline already does this: `format_training_example()`
calls `get_persona_instruction()` independently for each many-to-one variant, and that
function randomizes the persona frame, skeleton, and constraint set on every call. Verified:
5 calls for the same text produced 4 unique frame selections. No changes needed.

### Instruction Prefix Cleaning

Their pipeline strips mechanical prefixes from generated descriptions ("This passage
describes...", "In this excerpt..."). We should audit our training data for similar
mechanical patterns in the persona frame + constraint combinations. Any repeated
boilerplate that the model could latch onto as a shortcut instead of learning actual
style should be varied or removed.

### Prompt Phrasing Variation

Their 15 templates use different phrasings of the same intent: "Write in X's style",
"Channel X's voice", "Emulate X's prose style". Our frames are scenario-based ("You
are writing in a diary...") which is deeper, but the frame *introductions* are
structurally uniform ("You are writing/giving/making..."). Adding phrasing variation
to how frames are introduced (e.g., "Imagine you are...", "Picture this:...",
"The scene is...") could prevent attention collapse onto the "You are" pattern.

### Instruction Unmasking (Loss Over Instructions)

Standard SFT practice masks instruction tokens (sets targets to `-100`) so loss
is computed only on response tokens. [Shi et al. 2024 — Instruction Tuning With
Loss Over Instructions](https://arxiv.org/abs/2405.14394) and Sebastian Raschka's
practical tests found that NOT masking (computing loss on instruction + response)
performs slightly better on some tasks (~4 points improvement on Llama 3 eval).

For style transfer this is especially relevant because the persona frame in the
instruction carries stylistic signal. Computing loss on the instruction tokens
might reinforce the persona→style mapping more strongly. To enable in
LLaMA-Factory, add `train_on_prompt: true` to the yaml config.

**Status**: worth trying as an ablation if initial results disappoint. Risk is
unknown — could help or hurt depending on how much the persona frame repetition
dominates the loss signal.

### LLM-as-Judge Post-Training Evaluation

Eval loss catches loss-minimization issues but can't see whether style quality
is genuine vs memorized. Raschka's workbook uses Ollama with Llama 3 or phi3 as
an automated judge, scoring fine-tuned model outputs on a 0-100 scale against a
held-out test set.

For Howard Russell specifically, we could score on:
- Style coherence (is the Russell+Lovecraft blend consistent?)
- Content preservation (did the input semantics survive?)
- Blended feel (not just pure Russell or pure Lovecraft)
- Freedom from LLM-speak ("Moreover", "Furthermore", hedging)

This would catch cases where training loss is low but the model has memorized
phrases rather than learned the style. Currently we rely on manual inspection
and eval loss only.

## Adding a New Author

1. **Curate corpus** — 50k+ words of clean author prose
2. **Create snowflake topics** matching the content to be restyled
3. **Add persona frames** to `generate_flat_training.py` (narrative + conceptual)
4. **Generate training data**:
   ```bash
   python scripts/generate_flat_training.py \
       --corpus data/corpus/curated/author.txt \
       --author "Author Name" \
       --output data/training/author \
       --snowflake-topics data/training/author/snowflake_topics.py \
       --format llama_factory --skip-curation --workers 4
   ```
5. **Filter and split** (runs automatically for llama_factory; rerun with `python scripts/filter_training_data.py data/training/author/train.jsonl`)
6. **Create worldview file** in `prompts/` with EXACT same persona frames as training
7. **Configure LlamaFactory** yaml and dataset_info.json
8. **Train on RunPod** (see docs/runpod.md)
9. **Convert adapter** to MLX for local inference with `--train-config <yaml>` so the chat template is recorded (see docs/runpod.md)

## References

Key sources informing the techniques and hyperparameters in this guide:

**LoRA rank selection for literary style:**
- [The Effects of Rank, Epochs, and Learning Rate on Training Textual LoRAs](https://www.runpod.io/blog/effects-of-rank-epochs-learning-rate-textual-loras) — RunPod blog, Nabokov style transfer ablation study. Tested r=16/128/256/512, found rank 256 optimal ("huge jump in style"), 128 only mildly improved, 512 incoherent. **Primary source for our rank 256 decision.**
- [Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — Sebastian Raschka's independent tests on instruction tuning, also found r=256 optimal. Key insight: "the more diverse the tasks in the dataset, the larger the r should be."

**Training data structure and chunking:**
- [Gertrude Stein Style Training](https://muratcankoylan.com/projects/gertrude-stein-style-training/) — Muratcan Koylan's blog post on single-book style transfer. Key findings: 150-400w chunks with overlap outperform larger chunks ("style lives in transitions"), diverse instruction templates prevent attention collapse, base models outperform instruct models. **Primary source for our chunking strategy.**
- [muratcankoylan/book-training](https://github.com/muratcankoylan/book-training) — companion repository with the pipeline implementation, instruction templates, and validation tests.

**LoRA architecture:**
- [Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning](https://huggingface.co/blog/damjan-k/rslora) — HuggingFace blog on rsLoRA, explaining why standard alpha/rank scaling fails at high ranks and how sqrt(rank) scaling restores gradient flow.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — original LoRA paper.

**Style transfer research:**
- [Survey of Text Style Transfer](https://arxiv.org/abs/2407.16737) — comprehensive survey of style transfer techniques.
- [SAG: Style-Aligned Article Generation](https://arxiv.org/abs/2410.03137) — LLM+SLM collaboration for style transfer with content-style separation.

**Instruction tuning and training loop:**
- [Instruction Tuning With Loss Over Instructions (Shi et al. 2024)](https://arxiv.org/abs/2405.14394) — empirical finding that computing loss on full sequence (instruction + response) can outperform the standard masked-instruction approach.
- Sebastian Raschka, *Build a Large Language Model From Scratch* (Manning, 2024) — definitive reference for LLM fundamentals, fine-tuning, and LoRA. The "Test Yourself" companion workbook (in `docs/`) covers Appendices D (warmup/cosine decay/gradient clipping) and E (LoRA mechanics). Chapter 7 covers instruction fine-tuning and validates our format/approach.

**Blended author / style merging techniques:**
- [LoRA Soups](https://arxiv.org/abs/2410.13025) — CAT merging for composing multiple LoRA adapters (future work direction).
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) — NeurIPS 2024, sign conflict resolution for merging adapters.
- [X-LoRA: Mixture of Low-Rank Adapter Experts](https://arxiv.org/abs/2402.07148) — dynamic routing between multiple LoRA adapters at inference.
