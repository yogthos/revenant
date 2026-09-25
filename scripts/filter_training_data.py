#!/usr/bin/env python3
"""Filter generated training rows and split them for LlamaFactory.

Takes the raw train.jsonl written by generate_flat_training.py and writes
train.jsonl, val.jsonl and dataset_info.json into a LlamaFactory directory.

Rows are dropped when:
- the input still has an entity placeholder or starts with stray punctuation
- the input keeps too much of the target's distinctive vocabulary (lexical bleed)
- the output/input word ratio is too high, or the input is too short
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
LLAMA_FACTORY_COLUMNS = ("instruction", "input", "output")

# NLI label order for cross-encoder/nli-deberta-v3-*.
CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Row checks
# ---------------------------------------------------------------------------

def row_problem(row: dict, max_ratio: float = 2.0, min_input_words: int = 15) -> Optional[str]:
    """Why a row should be dropped, or None if it's fine."""
    from src.llm.mlx_provider import has_placeholder_residue
    from generate_flat_training import check_lexical_bleed

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
    ok, overlap = check_lexical_bleed(inp, out)
    if not ok:
        return f"lexical bleed {overlap:.0%}"
    return None


# ---------------------------------------------------------------------------
# Two-way entailment
# ---------------------------------------------------------------------------

def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3]


def _windows(sentences: List[str], max_words: int = 120) -> List[str]:
    """Overlapping premise windows short enough for the NLI model's context."""
    windows = []
    for start in range(len(sentences)):
        words = 0
        end = start
        while end < len(sentences) and (words == 0 or words + len(sentences[end].split()) <= max_words):
            words += len(sentences[end].split())
            end += 1
        windows.append(" ".join(sentences[start:end]))
        if end == len(sentences):
            break
    return windows or [" ".join(sentences)]


def _softmax(row) -> List[float]:
    exps = [math.exp(x - max(row)) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def _direction_problem(premise: str, hypothesis: str, nli, min_fraction: float,
                       max_contradiction: float) -> Optional[str]:
    hyps = _sentences(hypothesis)
    if not hyps:
        return None
    windows = _windows(_sentences(premise) or [premise])
    pairs = [(w, h) for h in hyps for w in windows]
    scores = [_softmax(list(r)) for r in nli.predict(pairs, show_progress_bar=False)]

    entailed = 0
    for i, hyp in enumerate(hyps):
        rows = scores[i * len(windows):(i + 1) * len(windows)]
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


def finalize(raw_path: Path, llama_factory_dir: Path, dataset_name: str,
             val_fraction: float = 0.05, nli: bool = True, seed: int = 42,
             block_size: int = 20, max_ratio: float = 2.0, min_input_words: int = 15,
             nli_min_fraction: float = 0.75, nli_model=None, log=print) -> dict:
    """Filter raw rows, split by source paragraph, write LlamaFactory files."""
    rows = list(_read_jsonl(raw_path))
    reasons: dict = {}

    def reject(reason: str) -> None:
        key = reason.split(" (")[0].split(":")[0]
        key = re.sub(r"\d+(\.\d+)?%?", "N", key)
        reasons[key] = reasons.get(key, 0) + 1

    kept = []
    for row in rows:
        problem = row_problem(row, max_ratio=max_ratio, min_input_words=min_input_words)
        if problem:
            reject(problem)
        else:
            kept.append(row)

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

    columns = {"prompt": "instruction", "query": "input", "response": "output"}
    info = {
        f"{dataset_name}_sft": {"file_name": "train.jsonl", "columns": columns},
        f"{dataset_name}_val": {"file_name": "val.jsonl", "columns": columns},
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
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--block-size", type=int, default=20,
                        help="Source paragraphs held out together (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-nli", action="store_true", help="Skip the two-way entailment filter")
    parser.add_argument("--nli-min-fraction", type=float, default=0.75,
                        help="Share of sentences that must be entailed in each direction")
    parser.add_argument("--max-ratio", type=float, default=2.0, help="Max output/input word ratio")
    parser.add_argument("--min-input-words", type=int, default=15)
    args = parser.parse_args()

    out_dir = args.llama_factory_dir or args.input.parent / "LlamaFactory"
    name = args.name or args.input.parent.name
    finalize(args.input, out_dir, name, val_fraction=args.val_fraction, nli=not args.no_nli,
             seed=args.seed, block_size=args.block_size, max_ratio=args.max_ratio,
             min_input_words=args.min_input_words, nli_min_fraction=args.nli_min_fraction)


if __name__ == "__main__":
    main()
