"""Sentence-structure comparison between a text and a rewrite of it.

Used to score restyled outputs (scripts/structure_score.py) and to reject
training inputs that keep the target's sentence order
(scripts/generate_flat_training.py).
"""

import re
import statistics
from typing import List, Optional

from .nlp import is_heading, split_into_paragraphs, split_into_sentences

# Both sides of a match must be mostly the same content words.
MATCH_THRESHOLD = 0.6

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "was", "were", "are", "but", "not", "any", "all",
    "its", "his", "her", "their", "them", "they", "from", "has", "have", "had", "which", "who",
    "what", "when", "where", "into", "than", "then", "there", "these", "those", "been", "being",
    "also", "will", "would", "can", "could", "should", "may", "might", "must", "our", "out",
    "about", "such", "only", "more", "most", "some", "one", "its", "it's", "upon", "very", "just",
}


def content_words(sentence: str) -> set:
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w
            for w in words if len(w) > 2 and w not in _STOPWORDS}


def kendall_tau(seq: List[int]) -> Optional[float]:
    pairs = [(a, b) for i, a in enumerate(seq) for b in seq[i + 1:]]
    if not pairs:
        return None
    score = sum((b > a) - (b < a) for a, b in pairs)
    return score / len(pairs)


def paragraph_score(inp: str, out: str) -> dict:
    in_sents = [content_words(s) for s in split_into_sentences(inp)]
    out_raw = split_into_sentences(out)
    out_sents = [content_words(s) for s in out_raw]

    one_to_one, followed = 0, []
    for o in out_sents:
        overlaps = [len(o & i) for i in in_sents]
        if not o or not any(overlaps):
            continue
        best = max(range(len(in_sents)), key=lambda k: overlaps[k])
        followed.append(best)
        shared = overlaps[best]
        if shared / len(o) >= MATCH_THRESHOLD and shared / len(in_sents[best]) >= MATCH_THRESHOLD:
            one_to_one += 1

    # Does the output open where the input did? The opener often carries the
    # transition from the previous paragraph.
    opener_kept = None
    if in_sents and in_sents[0] and out_sents and out_sents[0]:
        first, opener = out_sents[0], in_sents[0]
        opener_kept = (len(first & opener) / len(opener) >= 0.5
                       or (bool(followed) and followed[0] == 0 and bool(first & opener)))

    lengths = [len(s.split()) for s in out_raw]
    return {
        "opener_kept": opener_kept,
        "n_in": len(in_sents),
        "n_out": len(out_raw),
        "one_to_one_count": one_to_one,
        "one_to_one": one_to_one / len(out_raw) if out_raw else 0.0,
        "order": kendall_tau(followed),
        "lengths": lengths,
        "mean_len": statistics.mean(lengths) if lengths else 0.0,
        "sd_len": statistics.pstdev(lengths) if lengths else 0.0,
    }


def _body_paragraphs(text: str) -> List[str]:
    return [p for p in split_into_paragraphs(text) if not is_heading(p)]


def document_score(inp: str, out: str, corpus_index: Optional[set] = None, n: int = 8) -> dict:
    inp_paras, out_paras = _body_paragraphs(inp), _body_paragraphs(out)
    if len(inp_paras) != len(out_paras):
        raise ValueError(f"input has {len(inp_paras)} paragraphs, output {len(out_paras)}")
    scores = [paragraph_score(i, o) for i, o in zip(inp_paras, out_paras)]
    lengths = [k for s in scores for k in s["lengths"]]
    n_out = sum(s["n_out"] for s in scores)
    orders = [s["order"] for s in scores if s["order"] is not None]
    openers = [s["opener_kept"] for s in scores if s["opener_kept"] is not None]
    return {
        "opener_kept": sum(openers) / len(openers) if openers else None,
        "paragraphs": len(scores),
        "n_in": sum(s["n_in"] for s in scores),
        "n_out": n_out,
        "one_to_one": sum(s["one_to_one_count"] for s in scores) / n_out if n_out else 0.0,
        "order": statistics.mean(orders) if orders else None,
        "mean_len": statistics.mean(lengths) if lengths else 0.0,
        "sd_len": statistics.pstdev(lengths) if lengths else 0.0,
        "copied": max((longest_copied_run(p, corpus_index, n) for p in out_paras), default=0)
        if corpus_index is not None else None,
    }


def shuffle_sentences(text: str, rng, max_tau: float = 0.2, tries: int = 20) -> str:
    """The text's sentences in a new order, with Kendall tau at most ``max_tau`` if one is found.

    Of ``tries`` random orders the least ordered is kept. DIPPER (Krishna et al.
    2023) shuffled its inputs this way so the model had to learn the order.
    """
    sentences = split_into_sentences(text)
    if len(sentences) < 2:
        return text
    best_tau, best = 2.0, list(range(len(sentences)))
    for _ in range(tries):
        order = list(range(len(sentences)))
        rng.shuffle(order)
        tau = kendall_tau(order) or 0.0  # never None: there are at least two
        if tau < best_tau:
            best_tau, best = tau, order
        if tau <= max_tau:
            break
    return " ".join(sentences[i] for i in best)


# Runs this long shared with the author's corpus are copying, not style
# (Russell is in the base model's pretraining data too).
COPIED_RUN_FLAG = 12


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", text.lower())


def ngram_index(corpus: str, n: int = 8) -> set:
    """Every n-word sequence in the corpus, lowercased and without punctuation."""
    words = _words(corpus)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def longest_copied_run(text: str, index: set, n: int = 8) -> int:
    """Length in words of the longest run of ``text`` found verbatim in the indexed corpus."""
    words = _words(text)
    best = streak = 0
    for i in range(len(words) - n + 1):
        streak = streak + 1 if tuple(words[i:i + n]) in index else 0
        if streak:
            best = max(best, streak + n - 1)
    return best
