# Teaching the Adapter Structure, Not Just Vocabulary

Beads epic: `text-style-transfer-c95`.

## The problem

The Hemmingway-1 rank-256 Russell adapter (run 2, 2026-09-29) writes Russell's
vocabulary but keeps the input's sentences and their order when the content
is new. GPTZero labels that "Possible AI Paraphrasing": AI text reworded
sentence by sentence. On content close to Russell's own (a hunting tiger
described in plain prose) it rebuilds the sentences and passes as 96% human.

`scripts/structure_score.py` measures this. For each paragraph it reports:

- **1:1**: the share of output sentences that restate exactly one input sentence;
- **order**: the Kendall tau of the input sentences the output follows (1 = same order);
- sentence length mean and spread.

| Output | 1:1 | order | GPTZero |
|---|---|---|---|
| input/finance.md, checkpoint 1100, scale 1.0 | 49% | 0.85 | AI |
| input/finance.md, checkpoint 1100, scale 1.5 | 40% | 0.77 | AI |
| input/finance.md, checkpoint 1500, scale 1.0 | 42% | 0.90 | AI (50%) |
| tiger passage, checkpoint 1100 | 0% | 0.75 | 96% human |

Eval loss picked checkpoint 1100 and was no guide to this.

## Why: every row type keeps Russell's order

Here each training row's input is scored against its Russell target, sampling
120 rows per type:

| Row type | 1:1 | order | rows with order > 0.8 |
|---|---|---|---|
| standard, robustness, snowflake, perspective (~60% of rows) | 19-26% | 0.93-0.96 | 87-93% |
| llm_style explainer / explainer_polished | 6-7% | 0.78-0.79 | 58-61% |
| llm_style conversational / memo / punchy | 4-6% | 0.66-0.78 | 47-63% |

Neutralization paraphrases Russell but keeps his order, and the noise only
touches words. The llm_style rewrites break sentences apart but still walk
through the points in Russell's order. Every row therefore teaches "keep the
order of the ideas, fix the words", and that is what the model does.

## What the literature says

**Systems that learned to restructure made the input order uncopyable.**
- DIPPER trained on paragraph pairs from different translations of the same
  novel and shuffled the input sentences "to allow for the model to learn
  content re-ordering". Its strongest detector evasion came from high lexical
  and high order diversity together. Krishna et al., *Paraphrasing evades
  detectors of AI-generated text, but retrieval is an effective defense*,
  NeurIPS 2023, https://arxiv.org/abs/2303.13408
- STRAP (inverse paraphrasing) kept only paraphrase pairs with under 50%
  unigram/trigram overlap and at least 50% reordering by Kendall tau, cutting
  50M pairs to 75K. Without the filter, style accuracy dropped because the
  model copied its input. Krishna, Wieting & Iyyer, *Reformulating
  Unsupervised Style Transfer as Paraphrase Generation*, EMNLP 2020,
  https://arxiv.org/abs/2010.05700
- Sentence permutation is a standard corruption for learning document-level
  order: BART, Lewis et al., ACL 2020, https://arxiv.org/abs/1910.13461

**Detectors read structure.**
- Discourse structure separates human from machine text even after
  paraphrasing. Kim et al., *Threads of Subtlety*, ACL 2024,
  https://aclanthology.org/2024.acl-long.298/
- GPTZero has had an "AI paraphrased" class since November 2024. It is trained
  on simulated paraphrase attacks and uses "deeper semantic and structural
  signals". Its features are not published.
  https://gptzero.me/news/ai-paraphrasing-detection/

**Training strength and checkpoint choice.**
- A good LoRA learning rate is about 10x the full fine-tuning rate, roughly
  independent of rank ("LoRA Without Regret", Thinking Machines, 2025,
  https://thinkingmachines.ai/blog/lora/). LoRA learns less of the target
  domain than full fine-tuning (Biderman et al., TMLR 2024,
  https://arxiv.org/abs/2405.09673).
- On a reconstruction task, validation loss rewards copying. InstructGPT's
  validation loss overfit after one epoch while human preference kept
  improving (Ouyang et al., 2022, https://arxiv.org/abs/2203.02155).
- The old Qwen2.5-32B run that passed GPTZero used alpha/rank 2 and lr 1e-5,
  about 3.4x this run's update size, and was trained well past its eval-loss
  minimum.

**Chat versus base models.** Instruction-tuned models write less like humans
than base models and keep their own dense style even when prompted otherwise
(Reinhart et al., PNAS 2025, https://www.pnas.org/doi/10.1073/pnas.2422455122).
Aligned models are also narrower (West & Potts, COLM 2025,
https://arxiv.org/abs/2505.00047). This is indirect: the base-model run we
compare against also differs in data size and steps.

**Preference training after SFT.** For authorship transfer, a preference stage
beat SFT-only inverse paraphrasing (ASTRAPOP, Liu, Agarwal & May, 2024,
https://arxiv.org/abs/2403.08043); STAMP iterates it (NAACL 2025,
https://arxiv.org/abs/2406.11581). SPIN-style self-play uses the model's own
outputs as the rejected side (Chen et al., 2024,
https://arxiv.org/abs/2401.01335). DPO can flatten prose; the token-level FTPO
in Antislop removed most slop without that loss (Paech et al., ICLR 2026,
https://arxiv.org/abs/2510.15061).

**Content descriptions as inputs.** Per-author fine-tunes on "describe this
excerpt" -> original excerpt were flagged 0% by GPTZero, against 91% for
in-context prompting (Chakrabarty et al., 2025,
https://arxiv.org/abs/2510.13939). A prose description keeps the relations
between claims, which the claim lists we tried earlier lost, and has no
sentence skeleton to copy. That study used fiction.

**LLM tropes.** LLMs from different families share the same idiosyncrasies,
and professional editors remove them in similar ways (Chakrabarty, Laban & Wu,
CHI 2025, https://arxiv.org/abs/2409.14509). The paper gives a 7-category
taxonomy and the LAMP edit corpus.

Evidence gaps: no paper isolates sentence-shuffle augmentation for LLM
author-voice fine-tuning, and GPTZero's structural features are not public.

## Plan

1. **Break sentence order in training inputs** (`c95.1`)
   - llm_style prompts present the points in a different order, and rewrites
     whose order against the Russell target is above 0.5 are rejected (STRAP).
   - In a share of the Russell-structure rows, the input sentences are
     shuffled after the NLI check, so meaning is checked on the original order
     (DIPPER).
2. **Train harder** (`c95.2`): alpha 512, lr 1e-5, `train_on_prompt`, about 4
   epochs. Choose the checkpoint with `structure_score.py` on
   `input/finance.md` (low 1:1, low order), then confirm with GPTZero.
3. Later: a preference stage on the adapter's own copying outputs (`c95.3`),
   and a base-versus-chat ablation (`c95.4`).
4. Optional: a 15-25% slice of prose content-description inputs (`c95.5`).
