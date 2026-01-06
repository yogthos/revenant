"""Sentence splitter for controlling run-on sentences.

Splits overly long sentences at natural conjunction points (", and", ", but", ", for")
to prevent the "wall of text" effect from LoRA models that overfit to long sentence patterns.

Based on the approach in docs/sentence_control.md.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitStats:
    """Statistics from sentence splitting."""
    sentences_processed: int = 0
    sentences_split: int = 0
    total_splits: int = 0
    original_avg_length: float = 0.0
    result_avg_length: float = 0.0


@dataclass
class SentenceSplitterConfig:
    """Configuration for sentence splitting."""
    max_sentence_length: int = 50  # Words - split sentences longer than this
    min_clause_length: int = 15  # Minimum words between split points
    split_conjunctions: List[str] = field(default_factory=lambda: ["and", "but", "for", "yet", "so"])
    length_variance: float = 0.3  # Variance factor (0.3 = 70%-130% of max_sentence_length)


class SentenceSplitter:
    """Splits run-on sentences at natural conjunction points.

    Uses spaCy to identify coordinating conjunctions (CC) preceded by commas,
    then splits long sentences at these natural break points.
    """

    def __init__(self, config: SentenceSplitterConfig = None):
        """Initialize the sentence splitter.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or SentenceSplitterConfig()
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def _get_effective_max_length(self) -> int:
        """Get max sentence length with random variance.

        Prevents uniform "medium-medium-medium" rhythm by varying the
        threshold per sentence (70%-130% of base max_sentence_length).

        Returns:
            Effective max length for this sentence.
        """
        if self.config.length_variance <= 0:
            return self.config.max_sentence_length

        # Calculate variance range (e.g., 0.3 variance = 70%-130%)
        min_factor = 1.0 - self.config.length_variance
        max_factor = 1.0 + self.config.length_variance

        base = self.config.max_sentence_length
        return random.randint(
            int(base * min_factor),
            int(base * max_factor)
        )

    def split(self, text: str) -> Tuple[str, SplitStats]:
        """Split run-on sentences in text.

        Args:
            text: Text to process.

        Returns:
            Tuple of (processed_text, stats).
        """
        stats = SplitStats()

        if not text or not text.strip():
            return text, stats

        doc = self.nlp(text)
        result_sentences = []

        original_lengths = []
        result_lengths = []

        for sent in doc.sents:
            stats.sentences_processed += 1
            sent_words = len([t for t in sent if not t.is_space])
            original_lengths.append(sent_words)

            # If sentence is acceptable length, keep it
            # Use variance to prevent uniform "medium-medium-medium" rhythm
            effective_max = self._get_effective_max_length()
            if sent_words <= effective_max:
                result_sentences.append(sent.text)
                result_lengths.append(sent_words)
                continue

            # Sentence is too long - look for split points
            split_results = self._split_sentence(doc, sent)

            if len(split_results) > 1:
                stats.sentences_split += 1
                stats.total_splits += len(split_results) - 1
                result_sentences.extend(split_results)
                for s in split_results:
                    result_lengths.append(len(s.split()))
            else:
                # Couldn't find good split points, keep original
                result_sentences.append(sent.text)
                result_lengths.append(sent_words)

        # Calculate averages
        if original_lengths:
            stats.original_avg_length = sum(original_lengths) / len(original_lengths)
        if result_lengths:
            stats.result_avg_length = sum(result_lengths) / len(result_lengths)

        if stats.total_splits > 0:
            logger.debug(
                f"Sentence splitting: {stats.sentences_split} sentences split "
                f"({stats.total_splits} total splits), "
                f"avg length {stats.original_avg_length:.1f} -> {stats.result_avg_length:.1f} words"
            )

        return " ".join(result_sentences), stats

    def _split_sentence(self, doc, sent) -> List[str]:
        """Split a single sentence at conjunction points.

        Args:
            doc: Full spaCy doc (needed for token access).
            sent: Sentence span to split.

        Returns:
            List of sentence strings (may be single item if no splits found).
        """
        results = []
        last_start = sent.start

        for token in sent:
            # Look for coordinating conjunctions
            if token.pos_ != "CCONJ":
                continue

            # Check if it's one of our target conjunctions
            if token.text.lower() not in self.config.split_conjunctions:
                continue

            # Check minimum clause length from last split
            words_since_last = len([
                t for t in doc[last_start:token.i]
                if not t.is_space and not t.is_punct
            ])

            if words_since_last < self.config.min_clause_length:
                continue

            # Check for preceding comma (the ", and" pattern)
            if token.i > 0 and doc[token.i - 1].text == ",":
                # This is a valid split point
                # Take everything up to (but not including) the comma
                chunk = doc[last_start:token.i - 1]
                chunk_text = chunk.text.strip()

                if chunk_text:
                    # Ensure it ends with proper punctuation
                    if not chunk_text[-1] in ".!?":
                        chunk_text += "."
                    results.append(chunk_text)

                # Skip the conjunction, start new clause after it
                last_start = token.i + 1

        # Append the remainder
        if last_start < sent.end:
            chunk = doc[last_start:sent.end]
            chunk_text = chunk.text.strip()

            if chunk_text:
                # Capitalize first letter if this is a new sentence
                if results:  # Only capitalize if we actually split
                    chunk_text = self._capitalize_first(chunk_text)
                results.append(chunk_text)

        # If no splits were made, return original
        if not results:
            return [sent.text]

        return results

    def _capitalize_first(self, text: str) -> str:
        """Capitalize the first letter of text.

        Args:
            text: Text to capitalize.

        Returns:
            Text with first letter capitalized.
        """
        if not text:
            return text

        # Find first alphabetic character
        for i, char in enumerate(text):
            if char.isalpha():
                return text[:i] + char.upper() + text[i+1:]

        return text


# Module singleton
_splitter: SentenceSplitter = None


def get_sentence_splitter(config: SentenceSplitterConfig = None) -> SentenceSplitter:
    """Get or create singleton sentence splitter instance."""
    global _splitter
    if _splitter is None:
        _splitter = SentenceSplitter(config)
    return _splitter


def split_sentences(text: str, max_length: int = 50) -> str:
    """Convenience function to split sentences in text.

    Args:
        text: Text to process.
        max_length: Maximum sentence length in words.

    Returns:
        Processed text with long sentences split.
    """
    config = SentenceSplitterConfig(max_sentence_length=max_length)
    splitter = SentenceSplitter(config)
    result, _ = splitter.split(text)
    return result
