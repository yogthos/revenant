"""Input perturbation shared by training data generation and inference.

Training applies light noise to the neutral input so the model learns to
rebuild prose rather than copy it, and inference applies the same noise so
its inputs match. Both import from here; don't copy this logic elsewhere.

Noise never removes meaning. Only articles are ever dropped. Other changes
are typos and swaps from a small synonym map. Dropping adjectives,
intensifiers or words like "only", "never" and "always" taught the model to
fill gaps with invented detail, so none of that happens here.
"""

import random
import re

SYNONYMS = {
    "big": ["large", "huge", "great"],
    "small": ["little", "tiny", "minor"],
    "old": ["ancient", "aged", "elderly"],
    "new": ["fresh", "recent", "modern"],
    "good": ["fine", "nice", "great"],
    "bad": ["poor", "awful", "terrible"],
    "house": ["building", "home", "dwelling"],
    "said": ["stated", "spoke", "remarked"],
    "walked": ["went", "moved", "traveled"],
    "looked": ["appeared", "seemed", "gazed"],
    "very": ["quite", "rather", "extremely"],
    "really": ["truly", "actually", "indeed"],
}

# The only words noise may delete.
DROPPABLE = frozenset({"the", "a", "an"})

_WORD_RE = re.compile(r"^(\W*)([\w'’-]+)(\W*)$")


def _split(word: str):
    """Split a token into (leading punctuation, core, trailing punctuation)."""
    match = _WORD_RE.match(word)
    if not match:
        return "", word, ""
    return match.groups()


def _synonym(word: str) -> str:
    lead, core, trail = _split(word)
    options = SYNONYMS.get(core.lower())
    if not options:
        return word
    synonym = random.choice(options)
    if core[0].isupper():
        synonym = synonym.capitalize()
    return lead + synonym + trail


def _swap_typo(word: str) -> str:
    lead, core, trail = _split(word)
    if len(core) <= 3:
        return word
    i = random.randint(1, len(core) - 2)
    core = core[:i] + core[i + 1] + core[i] + core[i + 2:]
    return lead + core + trail


def _double_typo(word: str) -> str:
    lead, core, trail = _split(word)
    if len(core) <= 2:
        return word
    i = random.randint(0, len(core) - 1)
    return lead + core[:i] + core[i] + core[i:] + trail


def _is_droppable(word: str) -> bool:
    lead, core, trail = _split(word)
    # Keep articles that carry punctuation so sentence boundaries survive.
    return not lead and not trail and core.lower() in DROPPABLE


def perturb_text(text: str, perturbation_rate: float = 0.08) -> str:
    """Apply light noise: each word has ``perturbation_rate`` chance of a
    synonym swap (40%), an article drop (30%) or an adjacent-letter typo (30%).
    """
    result = []
    for word in text.split():
        if random.random() > perturbation_rate:
            result.append(word)
            continue

        choice = random.random()
        if choice < 0.4:
            result.append(_synonym(word))
        elif choice < 0.7:
            if not _is_droppable(word):
                result.append(word)
        else:
            result.append(_swap_typo(word))

    return " ".join(result)


def heavy_perturb_text(text: str, perturbation_rate: float = 0.15) -> str:
    """Heavier noise for robustness rows: synonym swaps, article drops,
    swap and double-letter typos, and case errors. Still never drops a
    word that isn't an article.
    """
    result = []
    for word in text.split():
        if random.random() > perturbation_rate:
            result.append(word)
            continue

        choice = random.random()
        if choice < 0.30:
            result.append(_synonym(word))
        elif choice < 0.50:
            if not _is_droppable(word):
                result.append(word)
        elif choice < 0.75:
            result.append(_swap_typo(word))
        elif choice < 0.90:
            result.append(_double_typo(word))
        elif random.random() < 0.5:
            result.append(word.lower())
        else:
            result.append(word.upper() if len(word) <= 4 else word)

    return " ".join(result)
