#!/usr/bin/env python3
"""Filter generated training rows, add the persona and split them for LlamaFactory.

Takes the raw train.jsonl written by generate_flat_training.py and writes
train.jsonl, val.jsonl and dataset_info.json into a LlamaFactory directory.

Each row gets the persona instruction inference builds
(src/persona/prompt_builder.build_persona_instruction): persona frame, word
count, the rhetorical skeleton of the most similar other corpus paragraph,
structural RAG hints and the constraints. It goes in the system turn and the
neutral text in the user turn, the way a chat model expects them.

Rows are dropped when:
- the input still has an entity placeholder or starts with stray punctuation
- the input keeps too much of the target's distinctive vocabulary (lexical bleed)
- the output/input word ratio is too high, or the input is too short
- the row won't fit in the training cutoff_len, which would cut off the output
  and its end-of-turn token
- (with NLI) the target states things the input lacks, or the input states
  things the target lacks, checked sentence by sentence in both directions

The split holds out whole source paragraphs, so overlapping chunks, many-to-one
variants and robustness copies of a validation paragraph never land in train.
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# LlamaFactory only needs these; the rest is generation metadata.
LLAMA_FACTORY_COLUMNS = ("system", "input", "output")
# Alpaca columns: persona in the system turn, neutral text as the user turn.
DATASET_COLUMNS = {"prompt": "input", "response": "output", "system": "system"}

# NLI label order for cross-encoder/nli-deberta-v3-*.
CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2


# Matches cutoff_len in the LlamaFactory yamls.
DEFAULT_MAX_TOKENS = 2048
# Qwen tokenizers average ~4.6 characters per token on this corpus and never
# went below 3.6, so 3.5 overestimates. The template adds about 20 tokens.
CHARS_PER_TOKEN = 3.5
TEMPLATE_TOKENS = 32


def word_count(text: str) -> int:
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Upper-bound token estimate that needs no tokenizer."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Row checks
# ---------------------------------------------------------------------------

def row_problem(row: dict, max_ratio: float = 2.0, min_input_words: int = 15,
                max_tokens: Optional[int] = DEFAULT_MAX_TOKENS) -> Optional[str]:
    """Why a row should be dropped, or None if it's fine."""
    from src.llm.mlx_provider import has_placeholder_residue
    from generate_flat_training import MAX_PHRASE_OVERLAP, check_lexical_bleed

    inp, out = row["input"], row["output"]
    if has_placeholder_residue(inp):
        return "entity placeholder in input"
    if re.match(r"^\s*[.,;:!?\-–—]", inp):
        return "input starts with punctuation"
    inp_words = word_count(inp)
    if inp_words < min_input_words:
        return f"input too short ({inp_words} words)"
    ratio = word_count(out) / max(inp_words, 1)
    if ratio > max_ratio:
        return f"output/input ratio {ratio:.1f}"
    ok, overlap = check_lexical_bleed(inp, out, max_phrase_overlap=MAX_PHRASE_OVERLAP)
    if not ok:
        from src.llm.mlx_provider import ngram_overlap
        if ngram_overlap(out, inp) > MAX_PHRASE_OVERLAP:
            return f"copied phrases {overlap:.0%}"
        return f"lexical bleed {overlap:.0%}"
    if max_tokens is not None:
        return length_problem(row, max_tokens)
    return None


def length_problem(row: dict, max_tokens: int = DEFAULT_MAX_TOKENS) -> Optional[str]:
    persona = row.get("system") or row.get("instruction", "")
    tokens = estimate_tokens(f"{persona}\n{row['input']}{row['output']}") + TEMPLATE_TOKENS
    if tokens > max_tokens:
        return f"too long for cutoff_len (~{tokens} tokens)"
    return None


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

def build_persona_instruction(*args, **kwargs):
    from src.persona.prompt_builder import build_persona_instruction as build
    return build(*args, **kwargs)


class PersonaBuilder:
    """Builds a row's persona the way inference builds it for a paragraph.

    Inference looks up guidance from the user's paragraph; the row's
    equivalent is its neutral input. The grafted skeleton never comes from the
    row's own paragraph, since inference never sees the target.
    """

    def __init__(self, worldview: str, rag=None, grafter=None):
        self.worldview = worldview
        self.rag = rag
        self.grafter = grafter

    def __call__(self, row: dict) -> str:
        inp, out = row["input"], row["output"]
        guidance = self.rag.get_guidance(inp).format_for_prompt() if self.rag else None
        graft = self.grafter.get_grafting_guidance(inp, exclude=out) if self.grafter else None
        return build_persona_instruction(
            inp,
            structural_guidance=guidance,
            grafting_guidance=graft,
            target_words=word_count(out),
            worldview=self.worldview,
        )


def load_persona_builder(author: str, worldview: str, rag: bool = True, grafting: bool = True) -> PersonaBuilder:
    """PersonaBuilder with the corpus index inference uses for ``author``."""
    from src.config import load_config
    from src.persona.prompt_builder import _load_persona_file

    frames = _load_persona_file(str(worldview))
    if not (frames["narrative_frames"] or frames["conceptual_frames"]):
        raise ValueError(f"No persona frames in {worldview}")

    config = load_config()
    structural_rag = grafter = None
    if rag or grafting:
        from src.rag.corpus_indexer import get_indexer
        if get_indexer().get_chunk_count(author) == 0:
            raise RuntimeError(
                f"No corpus indexed for {author!r}. Run: python scripts/load_corpus.py "
                f"--input <corpus> --author {author!r} --clear"
            )
    if rag:
        from src.rag.structural_rag import get_structural_rag
        structural_rag = get_structural_rag(author)
        structural_rag.load_patterns(sample_size=config.generation.rag_sample_size)
    if grafting:
        from src.llm.provider import create_critic_provider
        from src.rag.structural_grafter import get_structural_grafter
        grafter = get_structural_grafter(author, create_critic_provider(config.llm))
    return PersonaBuilder(str(worldview), rag=structural_rag, grafter=grafter)


# ---------------------------------------------------------------------------
# Two-way entailment
# ---------------------------------------------------------------------------

def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 3}


def _aligned_spans(premise_sents: List[str], hyp: str, top: int = 2, width: int = 1) -> List[str]:
    """Short premise spans around the sentences that best match ``hyp``.

    nli-deberta-v3-small misreads long multi-sentence premises (a paragraph
    can fail to entail its own first sentence), so each hypothesis sentence
    is checked against 1-3 sentence spans around its closest lexical matches.
    """
    words = _content_words(hyp)
    ranked = sorted(range(len(premise_sents)),
                    key=lambda i: -len(words & _content_words(premise_sents[i])))[:top]
    spans = []
    for i in ranked:
        for lo in range(max(0, i - width), i + 1):
            for hi in range(i + 1, min(len(premise_sents), i + width + 1) + 1):
                span = " ".join(premise_sents[lo:hi])
                if span not in spans:
                    spans.append(span)
    return spans


def _softmax(row) -> List[float]:
    exps = [math.exp(x - max(row)) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def _direction_problem(premise: str, hypothesis: str, nli, min_fraction: float,
                       max_contradiction: float) -> Optional[str]:
    hyps = _sentences(hypothesis)
    if not hyps:
        return None
    premise_sents = _sentences(premise) or [premise]
    spans = [_aligned_spans(premise_sents, h) for h in hyps]
    pairs = [(span, h) for h, hyp_spans in zip(hyps, spans) for span in hyp_spans]
    scores = [_softmax(list(r)) for r in nli.predict(pairs, show_progress_bar=False)]

    entailed = 0
    start = 0
    for hyp, hyp_spans in zip(hyps, spans):
        rows = scores[start:start + len(hyp_spans)]
        start += len(hyp_spans)
        best = max(r[ENTAILMENT] for r in rows)
        if best >= 0.5:
            entailed += 1
        elif min(r[CONTRADICTION] for r in rows) > max_contradiction:
            return f"contradiction: {hyp[:60]!r}"

    fraction = entailed / len(hyps)
    if fraction < min_fraction:
        return f"only {fraction:.0%} of sentences supported"
    return None


def entailment_problem(neutral: str, styled: str, nli, min_fraction: float = 0.75,
                       max_contradiction: float = 0.9) -> Optional[str]:
    """Check meaning is preserved both ways, sentence by sentence.

    Target sentences must be supported by the input (nothing added) and input
    sentences by the target (nothing dropped).
    """
    added = _direction_problem(neutral, styled, nli, min_fraction, max_contradiction)
    if added:
        return f"target adds content ({added})"
    dropped = _direction_problem(styled, neutral, nli, min_fraction, max_contradiction)
    if dropped:
        return f"target drops content ({dropped})"
    return None


def load_nli_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/nli-deberta-v3-small")


# ---------------------------------------------------------------------------
# Grouped split
# ---------------------------------------------------------------------------

def split_by_source(rows: List[dict], val_fraction: float = 0.05, seed: int = 42,
                    block_size: int = 20) -> Tuple[List[dict], List[dict], int]:
    """Split rows so no source paragraph appears in both train and val.

    Paragraphs are held out in contiguous blocks, since overlapping chunks
    span neighbouring paragraphs. A row whose paragraphs fall on both sides is
    dropped. Rows without source ids (old files) are grouped by output text.

    Returns (train, val, dropped_count).
    """
    ids = sorted({p for r in rows for p in r.get("source_paragraphs") or ()})
    blocks = sorted({p // block_size for p in ids})
    rng = random.Random(seed)
    n_val = max(1, round(len(blocks) * val_fraction)) if blocks else 0
    val_blocks = set(rng.sample(blocks, n_val)) if n_val else set()

    train, val, dropped = [], [], 0
    for row in rows:
        src = row.get("source_paragraphs") or ()
        if src:
            held = {p // block_size in val_blocks for p in src}
            if held == {True}:
                val.append(row)
            elif held == {False}:
                train.append(row)
            else:
                dropped += 1
        else:
            digest = hashlib.md5(f"{seed}:{row['output']}".encode()).digest()
            (val if digest[0] / 256 < val_fraction else train).append(row)
    return train, val, dropped


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row[k] for k in LLAMA_FACTORY_COLUMNS}, ensure_ascii=False) + "\n")


def finalize(raw_path: Path, llama_factory_dir: Path, dataset_name: str, *, persona,
             val_fraction: float = 0.05, nli: bool = True, seed: int = 42,
             block_size: int = 20, max_ratio: float = 2.0, min_input_words: int = 15,
             nli_min_fraction: float = 0.75, nli_model=None, max_tokens: int = DEFAULT_MAX_TOKENS,
             log=print) -> dict:
    """Filter raw rows, add the persona, split by source paragraph, write LlamaFactory files.

    ``persona`` maps a row to its system prompt (see PersonaBuilder).
    """
    rows = list(_read_jsonl(raw_path))
    reasons: dict = {}

    def reject(reason: str) -> None:
        key = reason.split(" (")[0].split(":")[0]
        key = re.sub(r"\d+(\.\d+)?%?", "N", key)
        reasons[key] = reasons.get(key, 0) + 1

    kept = []
    for i, row in enumerate(rows):
        problem = row_problem(row, max_ratio=max_ratio, min_input_words=min_input_words, max_tokens=None)
        if not problem:
            row["system"] = persona(row)
            problem = length_problem(row, max_tokens)
        if problem:
            reject(problem)
        else:
            kept.append(row)
        if (i + 1) % 1000 == 0:
            log(f"  Persona: {i + 1}/{len(rows)} rows, {len(kept)} kept")

    if nli:
        model = nli_model or load_nli_model()
        checked = []
        for i, row in enumerate(kept):
            problem = entailment_problem(row["input"], row["output"], model, min_fraction=nli_min_fraction)
            if problem:
                reject(problem)
            else:
                checked.append(row)
            if (i + 1) % 500 == 0:
                log(f"  NLI: {i + 1}/{len(kept)} checked, {len(checked)} kept")
        kept = checked

    train, val, straddling = split_by_source(kept, val_fraction=val_fraction, seed=seed, block_size=block_size)

    llama_factory_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(llama_factory_dir / "train.jsonl", train)
    _write_jsonl(llama_factory_dir / "val.jsonl", val)

    info = {
        f"{dataset_name}_sft": {"file_name": "train.jsonl", "columns": dict(DATASET_COLUMNS)},
        f"{dataset_name}_val": {"file_name": "val.jsonl", "columns": dict(DATASET_COLUMNS)},
    }
    (llama_factory_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")

    stats = {"raw": len(rows), "rejected": reasons, "straddling": straddling,
             "train": len(train), "val": len(val)}
    log(f"Rows: {len(rows)} raw -> {len(train)} train + {len(val)} val "
        f"({sum(reasons.values())} rejected, {straddling} straddled the split)")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log(f"  {count:6d}  {reason}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter training rows and write LlamaFactory train/val splits")
    parser.add_argument("input", type=Path, help="Raw train.jsonl from generate_flat_training.py")
    parser.add_argument("--llama-factory-dir", type=Path, default=None,
                        help="Output directory (default: <input dir>/LlamaFactory)")
    parser.add_argument("--name", default=None, help="Dataset name prefix (default: input directory name)")
    parser.add_argument("--author", required=True, help="Author name the corpus is indexed under")
    parser.add_argument("--worldview", required=True,
                        help="Persona file in prompts/ (the adapter's worldview in config.json)")
    parser.add_argument("--no-rag", action="store_true",
                        help="Leave structural RAG hints out (only if inference runs without them)")
    parser.add_argument("--no-grafting", action="store_true",
                        help="Leave grafted skeletons out (only if inference runs without them)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--block-size", type=int, default=20,
                        help="Source paragraphs held out together (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-nli", action="store_true", help="Skip the two-way entailment filter")
    parser.add_argument("--nli-min-fraction", type=float, default=0.75,
                        help="Share of sentences that must be entailed in each direction")
    parser.add_argument("--max-ratio", type=float, default=2.0, help="Max output/input word ratio")
    parser.add_argument("--min-input-words", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Drop rows longer than this (the yaml's cutoff_len)")
    args = parser.parse_args()

    out_dir = args.llama_factory_dir or args.input.parent / "LlamaFactory"
    name = args.name or args.input.parent.name
    persona = load_persona_builder(args.author, args.worldview, rag=not args.no_rag,
                                   grafting=not args.no_grafting)
    finalize(args.input, out_dir, name, persona=persona, val_fraction=args.val_fraction, nli=not args.no_nli,
             seed=args.seed, block_size=args.block_size, max_ratio=args.max_ratio,
             min_input_words=args.min_input_words, nli_min_fraction=args.nli_min_fraction,
             max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
