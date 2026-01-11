"""Content type classification for persona frame selection.

Classifies text as NARRATIVE or CONCEPTUAL to select appropriate persona frames.
Used by both training (generate_flat_training.py) and inference (prompt_builder.py).

CRITICAL: This module is shared between training and inference. Any changes here
affect both. The classification logic must remain consistent to match training.
"""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc


class ContentType(Enum):
    """Type of content for prompt selection."""
    NARRATIVE = "narrative"      # Stories, events, characters, sequences
    CONCEPTUAL = "conceptual"    # Ideas, explanations, mechanisms, definitions


def classify_content_type(text: str, default_to_narrative: bool = True) -> ContentType:
    """Classify text as NARRATIVE or CONCEPTUAL.

    Uses spaCy for robust detection matching training exactly.

    NARRATIVE indicators:
    - Named entities (PERSON, GPE, LOC, FAC)
    - Past tense verbs (VBD, VBN)
    - Temporal markers (then, after, before, when)
    - Sequence words (first, next, finally)

    CONCEPTUAL indicators:
    - Abstract nouns (concept, theory, mechanism, process)
    - Definition patterns (X is defined as, X refers to)
    - Present tense generalizations
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

    narrative_score = 0
    conceptual_score = 0

    # Check for named entities (strong narrative signal)
    person_entities = sum(1 for ent in doc.ents if ent.label_ in ['PERSON', 'GPE', 'LOC', 'FAC'])
    if person_entities >= 2:
        narrative_score += 3
    elif person_entities >= 1:
        narrative_score += 1

    # Check for temporal markers
    temporal_markers = {'then', 'after', 'before', 'when', 'while', 'during', 'later',
                       'earlier', 'soon', 'finally', 'eventually', 'suddenly', 'once'}
    text_lower = text.lower()
    temporal_count = sum(1 for marker in temporal_markers if f' {marker} ' in f' {text_lower} ')
    narrative_score += min(temporal_count, 3)

    # Check for past tense verbs (narrative signal)
    past_tense_count = sum(1 for token in doc if token.tag_ in ['VBD', 'VBN'])
    if past_tense_count >= 5:
        narrative_score += 2
    elif past_tense_count >= 2:
        narrative_score += 1

    # Check for sequence words
    sequence_words = {'first', 'second', 'third', 'next', 'finally', 'began', 'started', 'ended'}
    sequence_count = sum(1 for word in sequence_words if word in text_lower)
    narrative_score += min(sequence_count, 2)

    # Check for abstract/conceptual vocabulary
    conceptual_words = {'concept', 'theory', 'mechanism', 'process', 'system', 'principle',
                       'function', 'method', 'approach', 'technique', 'structure', 'pattern',
                       'relationship', 'connection', 'effect', 'cause', 'result', 'factor',
                       'element', 'component', 'aspect', 'nature', 'essence', 'phenomenon'}
    conceptual_count = sum(1 for word in conceptual_words if word in text_lower)
    conceptual_score += min(conceptual_count * 2, 4)

    # Check for definition patterns
    definition_patterns = ['is defined as', 'refers to', 'means that', 'is called',
                          'can be described as', 'is characterized by', 'consists of']
    if any(pattern in text_lower for pattern in definition_patterns):
        conceptual_score += 3

    # Check for impersonal/generalizing constructions
    if text_lower.startswith(('it is', 'there are', 'there is', 'this is', 'these are')):
        conceptual_score += 1

    # Present tense generalizations (weaker signal)
    present_tense_count = sum(1 for token in doc if token.tag_ in ['VBZ', 'VBP'] and token.dep_ == 'ROOT')
    conceptual_score += min(present_tense_count, 2)

    # Return classification with hysteresis to avoid flipping on edge cases
    if narrative_score > conceptual_score + 1:
        return ContentType.NARRATIVE
    elif conceptual_score > narrative_score + 1:
        return ContentType.CONCEPTUAL
    else:
        # Tied or close - use default
        return ContentType.NARRATIVE if default_to_narrative else ContentType.CONCEPTUAL


def is_narrative(text: str) -> bool:
    """Convenience function: returns True if text is narrative, False if conceptual."""
    return classify_content_type(text) == ContentType.NARRATIVE
