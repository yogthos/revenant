"""Content type classification for persona frame selection.

Classifies text as NARRATIVE or CONCEPTUAL to select appropriate persona frames.
Used by both training (generate_flat_training.py) and inference (prompt_builder.py).

CRITICAL: This module is shared between training and inference. Any changes here
affect both. The classification logic must remain consistent to match training.
"""

from enum import Enum
from typing import TYPE_CHECKING

from .logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from spacy.tokens import Doc


# Weight of the main-clause tense, the strongest signal. The other signals are
# counts capped at a few points, so on a long paragraph they no longer outvote it.
TENSE_WEIGHT = 6


def _main_clause_tense_ratio(doc) -> float:
    """Share of sentences whose main verb is past tense, or 0.5 if none are finite.

    Looks at the root verb and its auxiliaries, so "was invented" is past and
    "is contained" is present. Counting past participles instead made every
    long expository paragraph look narrative.
    """
    past = present = 0
    for sent in doc.sents:
        root = sent.root
        verbs = [root] + [c for c in root.children if c.dep_ in ("aux", "auxpass")]
        tenses = {t for v in verbs if "Fin" in v.morph.get("VerbForm") or v.tag_ in ("VBD", "VBZ", "VBP", "MD")
                  for t in (v.morph.get("Tense") or (["Pres"] if v.tag_ == "MD" else []))}
        if "Past" in tenses:
            past += 1
        elif "Pres" in tenses:
            present += 1
    return past / (past + present) if past + present else 0.5


class ContentType(Enum):
    """Type of content for prompt selection."""
    NARRATIVE = "narrative"      # Stories, events, characters, sequences
    CONCEPTUAL = "conceptual"    # Ideas, explanations, mechanisms, definitions


def classify_content_type(text: str, default_to_narrative: bool = True) -> ContentType:
    """Classify text as NARRATIVE or CONCEPTUAL.

    Uses spaCy for robust detection matching training exactly.

    NARRATIVE indicators:
    - Main clauses in the past tense (the strongest signal)
    - Named entities (PERSON, GPE, LOC, FAC)
    - Temporal markers (then, after, before, when)
    - Sequence words (first, next, finally)

    CONCEPTUAL indicators:
    - Abstract nouns (concept, theory, mechanism, process)
    - Definition patterns (X is defined as, X refers to)
    - Main clauses in the present tense
    - Impersonal constructions (it is, there are)

    Args:
        text: Text to classify.
        default_to_narrative: If scores are tied, return NARRATIVE (for fiction authors).

    Returns:
        ContentType.NARRATIVE or ContentType.CONCEPTUAL
    """
    from .nlp import get_nlp

    nlp = get_nlp()
    doc = nlp(text)

    narrative_score = 0.0
    conceptual_score = 0.0

    # Main-clause tense: stories are told in the past, arguments in the present
    past_ratio = _main_clause_tense_ratio(doc)
    narrative_score += TENSE_WEIGHT * past_ratio
    conceptual_score += TENSE_WEIGHT * (1 - past_ratio)

    # Named people and places (weak: essays name people too)
    person_entities = sum(1 for ent in doc.ents if ent.label_ in ['PERSON', 'GPE', 'LOC', 'FAC'])
    narrative_score += min(person_entities, 2)

    # Temporal markers. "when", "while" and "once" are left out: arguments use them as much as stories.
    temporal_markers = {'then', 'after', 'before', 'during', 'later', 'earlier', 'soon', 'finally',
                        'eventually', 'suddenly'}
    text_lower = text.lower()
    # Use word boundary check: split into words and strip punctuation
    text_words = {w.strip('.,!?;:"\'-') for w in text_lower.split()}
    temporal_count = sum(1 for marker in temporal_markers if marker in text_words)
    narrative_score += min(temporal_count, 3)

    # Check for sequence words (use text_words for word-boundary matching)
    sequence_words = {'first', 'second', 'third', 'next', 'finally', 'began', 'started', 'ended'}
    sequence_count = sum(1 for word in sequence_words if word in text_words)
    narrative_score += min(sequence_count, 2)

    # Check for abstract/conceptual vocabulary (use text_words for word-boundary matching)
    conceptual_words = {'concept', 'theory', 'mechanism', 'process', 'system', 'principle',
                       'function', 'method', 'approach', 'technique', 'structure', 'pattern',
                       'relationship', 'connection', 'effect', 'cause', 'result', 'factor',
                       'element', 'component', 'aspect', 'nature', 'essence', 'phenomenon'}
    conceptual_count = sum(1 for word in conceptual_words if word in text_words)
    conceptual_score += min(conceptual_count * 2, 4)

    # Check for definition patterns
    definition_patterns = ['is defined as', 'refers to', 'means that', 'is called',
                          'can be described as', 'is characterized by', 'consists of']
    if any(pattern in text_lower for pattern in definition_patterns):
        conceptual_score += 3

    # Check for impersonal/generalizing constructions
    if text_lower.startswith(('it is', 'there are', 'there is', 'this is', 'these are')):
        conceptual_score += 1

    # Return classification with hysteresis to avoid flipping on edge cases
    if narrative_score > conceptual_score + 1:
        result = ContentType.NARRATIVE
    elif conceptual_score > narrative_score + 1:
        result = ContentType.CONCEPTUAL
    else:
        # Close: go with the tense, or the default when there are no finite verbs
        if past_ratio != 0.5:
            result = ContentType.NARRATIVE if past_ratio > 0.5 else ContentType.CONCEPTUAL
        else:
            result = ContentType.NARRATIVE if default_to_narrative else ContentType.CONCEPTUAL
        logger.debug(
            f"Borderline classification: narrative={narrative_score:.1f}, conceptual={conceptual_score:.1f} "
            f"→ {result.value}. Text: {text[:80]}..."
        )

    return result


def is_narrative(text: str) -> bool:
    """Convenience function: returns True if text is narrative, False if conceptual."""
    return classify_content_type(text) == ContentType.NARRATIVE
