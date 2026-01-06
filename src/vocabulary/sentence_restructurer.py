"""Sentence restructurer for converting mechanical patterns to organic ones.

Transforms:
- Balanced "A, and B" openings → Inverted/periodic structures
- Subject-first sentences → Prepositional openings
- Linear progressions → Interrupted syntax

This is a heuristic-based post-processor that applies structural transformations
while preserving semantic content.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RestructureStats:
    """Statistics from restructuring."""
    sentences_analyzed: int = 0
    balanced_detected: int = 0
    inversions_applied: int = 0
    interruptions_added: int = 0


class SentenceRestructurer:
    """Restructures mechanical sentence patterns into organic ones."""

    # Preposition phrases for inverted openings
    INVERSION_PREFIXES = [
        "From beyond the bounds of",
        "In the depths of",
        "Through the veil of",
        "Within the shadow of",
        "Beneath the weight of",
        "Amidst the darkness of",
        "Beyond the pale of",
        "Upon the threshold of",
    ]

    # Patterns for detecting balanced openings (first 50 chars)
    BALANCED_PATTERNS = [
        r'^(The|A|An|This|That) \w+ (is|was|has|had|are|were) [\w\s]+, and ',
        r'^(It) (is|was) [\w\s]+, and ',
        r'^[\w\s]+ (is|are|was|were) [\w\s]+, and (the|a|this|that|it)',
    ]

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def restructure(self, text: str) -> Tuple[str, RestructureStats]:
        """Restructure mechanical patterns in text.

        Args:
            text: Text to restructure.

        Returns:
            Tuple of (restructured_text, stats).
        """
        stats = RestructureStats()

        if not text or not text.strip():
            return text, stats

        sentences = split_into_sentences(text)
        result_sentences = []

        for i, sent in enumerate(sentences):
            stats.sentences_analyzed += 1

            # Check for balanced opening (only on first few sentences)
            if i < 3 and self._has_balanced_opening(sent):
                stats.balanced_detected += 1
                restructured = self._invert_balanced_opening(sent)
                if restructured != sent:
                    stats.inversions_applied += 1
                    result_sentences.append(restructured)
                    continue

            # Check if we should add an interruption
            if len(sent.split()) > 20 and '—' not in sent and i > 0:
                interrupted = self._add_interruption(sent)
                if interrupted != sent:
                    stats.interruptions_added += 1
                    result_sentences.append(interrupted)
                    continue

            result_sentences.append(sent)

        result = ' '.join(result_sentences)

        if stats.inversions_applied > 0 or stats.interruptions_added > 0:
            logger.debug(
                f"Restructured: {stats.inversions_applied} inversions, "
                f"{stats.interruptions_added} interruptions"
            )

        return result, stats

    def _has_balanced_opening(self, sent: str) -> bool:
        """Check if sentence has a balanced 'A, and B' opening."""
        first_part = sent[:100]  # Expanded from 60 to catch longer clauses

        for pattern in self.BALANCED_PATTERNS:
            if re.match(pattern, first_part, re.IGNORECASE):
                return True

        # Also check for ", and" in first 80 chars with short pre-clause
        if ', and ' in first_part[:80]:
            before_and = first_part.split(', and')[0]
            words_before = len(before_and.split())
            if words_before <= 12:  # Expanded from 10 to catch more cases
                return True

        return False

    def _invert_balanced_opening(self, sent: str) -> str:
        """Convert balanced opening to inverted structure.

        Example:
            Input: "The universe is bounded, and this tenet of our experience..."
            Output: "From beyond mortal comprehension, the universe is bounded—and this tenet..."
        """
        # Parse the sentence
        doc = self.nlp(sent)

        # Find the first subject (nsubj or nsubjpass) and main verb
        subject = None
        main_verb = None
        for token in doc:
            # Look for subject (including passive subjects)
            if token.dep_ in ("nsubj", "nsubjpass") and subject is None:
                # Only take subject from first clause (before "and")
                if token.i < 10:  # Must be early in sentence
                    # Get the full subject phrase
                    subject_tokens = [t for t in token.subtree if t.i < 10]
                    subject = ' '.join(t.text for t in sorted(subject_tokens, key=lambda t: t.i))
            if token.dep_ == "ROOT":
                main_verb = token

        if not subject or not main_verb:
            return sent  # Can't restructure

        # Extract key nouns for the inversion prefix
        nouns = [t.text.lower() for t in doc if t.pos_ == "NOUN"]

        # Choose appropriate prefix based on content
        prefix = self._choose_inversion_prefix(nouns, sent)

        # Try to restructure
        # Find where ", and" occurs
        and_pos = sent.find(', and ')
        if and_pos == -1:
            return sent

        first_clause = sent[:and_pos]
        rest = sent[and_pos + 2:]  # Keep ", and" as "—and"

        # Build the inverted sentence
        # Remove the subject from the beginning and put it after the prefix
        if first_clause.lower().startswith(subject.lower()):
            after_subject = first_clause[len(subject):].strip()
            inverted = f"{prefix} {subject.lower()} {after_subject}—{rest.strip()}"
            # Capitalize first letter
            inverted = inverted[0].upper() + inverted[1:]
            return inverted

        return sent

    def _choose_inversion_prefix(self, nouns: List[str], sent: str) -> str:
        """Choose an appropriate inversion prefix based on content."""
        import random

        # Map content to appropriate prefixes
        cosmic_words = {'universe', 'cosmos', 'space', 'void', 'abyss', 'infinity'}
        limit_words = {'limit', 'boundary', 'edge', 'border', 'bound'}
        experience_words = {'experience', 'knowledge', 'understanding', 'comprehension'}
        dark_words = {'dark', 'shadow', 'night', 'black'}

        nouns_set = set(nouns)
        sent_lower = sent.lower()

        if nouns_set & cosmic_words or 'universe' in sent_lower:
            return random.choice([
                "From beyond the bounds of mortal comprehension,",
                "In the vast and terrible expanse where",
                "Beyond the furthest reaches of human understanding,",
            ])

        if nouns_set & limit_words or 'limit' in sent_lower or 'bound' in sent_lower:
            return random.choice([
                "At the very edge of conceivable reality,",
                "Upon the threshold of the unknowable,",
                "Where comprehension fails and reason falters,",
            ])

        if nouns_set & experience_words:
            return random.choice([
                "Through the lens of our feeble understanding,",
                "In the dim light of human perception,",
                "Filtered through the inadequacy of mortal experience,",
            ])

        if nouns_set & dark_words:
            return random.choice([
                "In the depths of that impenetrable darkness,",
                "Amidst the shadows that harbor nameless things,",
                "Within the blackness where light fears to tread,",
            ])

        # Default cosmic horror prefixes
        return random.choice(self.INVERSION_PREFIXES)

    def _add_interruption(self, sent: str) -> str:
        """Add an em-dash interruption to a long sentence.

        Example:
            Input: "The debt which was estimated at $4.5 trillion posed a risk."
            Output: "The debt—a staggering sum of $4.5 trillion—posed a risk."
        """
        doc = self.nlp(sent)

        # Look for "which was/is" or "that was/is" patterns
        for i, token in enumerate(doc):
            if token.text.lower() in ('which', 'that') and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.lemma_ == 'be':
                    # Found "which was/is" - convert to em-dash interruption
                    # Find the end of the relative clause
                    clause_end = self._find_clause_end(doc, i)
                    if clause_end > i + 3:  # Meaningful clause
                        before = doc[:i].text
                        clause_content = doc[i + 2:clause_end].text  # Skip "which was"
                        after = doc[clause_end:].text

                        # Add em-dash interruption
                        return f"{before}—{clause_content}—{after}"

        # Look for ", ADJ NOUN," patterns that could become em-dash
        comma_positions = [i for i, t in enumerate(doc) if t.text == ',']
        if len(comma_positions) >= 2:
            # Try converting a parenthetical to em-dash
            start = comma_positions[0]
            end = comma_positions[1]
            if 2 <= end - start <= 6:  # Short parenthetical
                before = doc[:start].text
                middle = doc[start + 1:end].text
                after = doc[end + 1:].text

                return f"{before}—{middle}—{after}"

        return sent

    def _find_clause_end(self, doc, start_idx: int) -> int:
        """Find the end of a relative clause."""
        depth = 0
        for i in range(start_idx, len(doc)):
            token = doc[i]
            if token.text in ('(', '['):
                depth += 1
            elif token.text in (')', ']'):
                depth -= 1
            elif depth == 0 and token.pos_ == "VERB" and token.dep_ == "ROOT":
                return i
            elif depth == 0 and token.text == ',':
                return i
        return len(doc)


# Module singleton
_restructurer: SentenceRestructurer = None


def get_sentence_restructurer() -> SentenceRestructurer:
    """Get or create singleton restructurer instance."""
    global _restructurer
    if _restructurer is None:
        _restructurer = SentenceRestructurer()
    return _restructurer


def restructure_sentences(text: str) -> str:
    """Convenience function to restructure sentences."""
    restructurer = get_sentence_restructurer()
    result, _ = restructurer.restructure(text)
    return result
