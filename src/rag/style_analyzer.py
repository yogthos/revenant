"""Style metrics analyzer for RAG retrieval.

Extracts structural style metrics from text chunks using spaCy.
"""

import re
from dataclasses import dataclass
from typing import List

from ..utils.logging import get_logger
from ..utils.nlp import (
    get_nlp,
    split_into_sentences,
    get_dependency_depth,
    get_pos_distribution,
    count_words,
)

logger = get_logger(__name__)


@dataclass
class StyleMetrics:
    """Style metrics for a text chunk."""

    avg_sentence_length: float
    sentence_length_std: float
    dependency_depth: float
    adjective_ratio: float
    verb_ratio: float
    punctuation_density: float
    avg_word_length: float

    def to_dict(self) -> dict:
        """Convert to dictionary for ChromaDB metadata."""
        return {
            "avg_sentence_length": self.avg_sentence_length,
            "sentence_length_std": self.sentence_length_std,
            "dependency_depth": self.dependency_depth,
            "adjective_ratio": self.adjective_ratio,
            "verb_ratio": self.verb_ratio,
            "punctuation_density": self.punctuation_density,
            "avg_word_length": self.avg_word_length,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StyleMetrics":
        """Create from dictionary."""
        return cls(
            avg_sentence_length=d.get("avg_sentence_length", 0.0),
            sentence_length_std=d.get("sentence_length_std", 0.0),
            dependency_depth=d.get("dependency_depth", 0.0),
            adjective_ratio=d.get("adjective_ratio", 0.0),
            verb_ratio=d.get("verb_ratio", 0.0),
            punctuation_density=d.get("punctuation_density", 0.0),
            avg_word_length=d.get("avg_word_length", 0.0),
        )

    def structural_distance(self, other: "StyleMetrics") -> float:
        """Calculate structural distance from another StyleMetrics.

        Lower values = more similar structure.
        """
        # Weighted distance across metrics
        weights = {
            "avg_sentence_length": 0.25,
            "sentence_length_std": 0.15,
            "dependency_depth": 0.20,
            "adjective_ratio": 0.15,
            "verb_ratio": 0.10,
            "punctuation_density": 0.10,
            "avg_word_length": 0.05,
        }

        total = 0.0
        for field, weight in weights.items():
            v1 = getattr(self, field)
            v2 = getattr(other, field)
            # Normalize difference by max value to avoid scale issues
            max_val = max(abs(v1), abs(v2), 1.0)
            diff = abs(v1 - v2) / max_val
            total += weight * diff

        return total


class StyleAnalyzer:
    """Analyzes text for style metrics."""

    def __init__(self):
        """Initialize the analyzer."""
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def analyze(self, text: str) -> StyleMetrics:
        """Analyze text and return style metrics.

        Args:
            text: Text to analyze.

        Returns:
            StyleMetrics with extracted style features.
        """
        if not text or not text.strip():
            return StyleMetrics(
                avg_sentence_length=0.0,
                sentence_length_std=0.0,
                dependency_depth=0.0,
                adjective_ratio=0.0,
                verb_ratio=0.0,
                punctuation_density=0.0,
                avg_word_length=0.0,
            )

        # Get sentences
        sentences = split_into_sentences(text)

        # Sentence length stats
        if sentences:
            lengths = [count_words(s) for s in sentences]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            length_std = variance ** 0.5
        else:
            avg_length = 0.0
            length_std = 0.0

        # Dependency depth
        depth = get_dependency_depth(text)

        # POS ratios
        pos_dist = get_pos_distribution(text)
        total_tokens = sum(pos_dist.values()) or 1
        adj_ratio = pos_dist.get("ADJ", 0) / total_tokens
        verb_ratio = pos_dist.get("VERB", 0) / total_tokens

        # Punctuation density (per 100 words)
        word_count = count_words(text) or 1
        punct_chars = len(re.findall(r'[;:—\-–,]', text))
        punct_density = (punct_chars / word_count) * 100

        # Average word length
        words = text.split()
        if words:
            avg_word_len = sum(len(w.strip('.,;:!?"\'()-—')) for w in words) / len(words)
        else:
            avg_word_len = 0.0

        return StyleMetrics(
            avg_sentence_length=avg_length,
            sentence_length_std=length_std,
            dependency_depth=depth,
            adjective_ratio=adj_ratio,
            verb_ratio=verb_ratio,
            punctuation_density=punct_density,
            avg_word_length=avg_word_len,
        )

    def analyze_batch(self, texts: List[str]) -> List[StyleMetrics]:
        """Analyze multiple texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of StyleMetrics.
        """
        return [self.analyze(text) for text in texts]


# Module-level singleton for convenience
_analyzer = None


def get_style_analyzer() -> StyleAnalyzer:
    """Get the singleton style analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = StyleAnalyzer()
    return _analyzer


def analyze_style(text: str) -> StyleMetrics:
    """Convenience function to analyze style metrics.

    Args:
        text: Text to analyze.

    Returns:
        StyleMetrics for the text.
    """
    return get_style_analyzer().analyze(text)
