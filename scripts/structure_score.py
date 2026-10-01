#!/usr/bin/env python3
"""How much of the input's sentence structure a restyled output kept.

GPTZero flags AI text reworded sentence by sentence as "AI paraphrasing", so
eval loss is a poor guide to which checkpoint to use. This compares an input
document with its restyled output, paragraph by paragraph:

  one_to_one  share of output sentences that restate exactly one input
              sentence (merged, split or rebuilt sentences don't count)
  order       Kendall tau of the input sentences the output follows
              (1 = same order, -1 = reversed)
  mean_len    mean output sentence length in words, and sd_len its spread
  opener      share of paragraphs whose output opens with the input's first
              sentence (where transitions from the previous paragraph sit)
  copied      with --corpus, the longest run of words copied verbatim from
              the author's corpus; COPIED_RUN_FLAG or more is marked "!"

Usage:
    python scripts/structure_score.py input/finance.md output/finance_*.md \
        --corpus data/corpus/curated/russell.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.structure import COPIED_RUN_FLAG, document_score, ngram_index, paragraph_score  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Score how much input sentence structure outputs kept")
    parser.add_argument("input", type=Path)
    parser.add_argument("outputs", type=Path, nargs="+")
    parser.add_argument("--corpus", type=Path, help="Author corpus, to flag text copied from it")
    args = parser.parse_args()

    inp = args.input.read_text()
    index = ngram_index(args.corpus.read_text()) if args.corpus else None
    base = document_score(inp, inp)
    print(f"{'file':40} {'sents':>9} {'1:1':>5} {'order':>6} {'len':>5} {'sd':>5} {'opener':>6}"
          + (f" {'copied':>7}" if index else ""))
    print(f"{'(input)':40} {base['n_in']:>9} {'':>5} {'':>6} {base['mean_len']:5.1f} {base['sd_len']:5.1f}")
    for path in args.outputs:
        s = document_score(inp, path.read_text(), corpus_index=index)
        order = f"{s['order']:6.2f}" if s["order"] is not None else f"{'-':>6}"
        line = (f"{str(path):40} {s['n_in']:>4}->{s['n_out']:<4} {s['one_to_one']:5.0%} {order} "
                f"{s['mean_len']:5.1f} {s['sd_len']:5.1f} {s['opener_kept']:6.0%}")
        if index:
            flag = "!" if s["copied"] >= COPIED_RUN_FLAG else " "
            line += f" {s['copied']:>6}{flag}"
        print(line)


if __name__ == "__main__":
    main()
