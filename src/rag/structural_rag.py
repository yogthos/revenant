"""Structural RAG - retrieves rhythm and syntax patterns, not content.

Instead of returning text examples that get copied, this returns:
- Rhythm fingerprints (sentence length sequences)
- Punctuation patterns
- Syntactic templates (POS sequences)
- Opening/transition patterns

This addresses the core issues:
- Mechanical Transitions → Natural transition rhythm from author
- Impersonal Tone → Emotional punctuation patterns (!, —, ...)
- Mechanical Precision → High variance in sentence lengths
- Formulaic Flow → Actual author rhythm fingerprints
- Robotic Formality → Fragment usage, informal punctuation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import random

from .structural_analyzer import (
    StructuralAnalyzer,
    RhythmFingerprint,
    StructuralStyle,
    get_structural_analyzer,
)
from .corpus_indexer import get_indexer
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StructuralGuidance:
    """Structural guidance for prompt injection."""
    rhythm_pattern: str  # e.g., "LONG → SHORT → MEDIUM → FRAGMENT → LONG"
    punctuation_hints: List[str]  # e.g., ["use dashes", "semicolons for lists"]
    length_guidance: str  # e.g., "Vary between 5 and 35 words per sentence"
    fragment_hint: str  # e.g., "Include 1-2 sentence fragments"
    opening_hint: str  # e.g., "Start with: determiner, noun, or adverb"

    def format_for_prompt(self) -> str:
        """Format as prompt injection."""
        lines = []

        if self.rhythm_pattern:
            lines.append(f"RHYTHM PATTERN: {self.rhythm_pattern}")

        if self.length_guidance:
            lines.append(f"LENGTH: {self.length_guidance}")

        if self.fragment_hint:
            lines.append(f"FRAGMENTS: {self.fragment_hint}")

        if self.punctuation_hints:
            hints = "; ".join(self.punctuation_hints)
            lines.append(f"PUNCTUATION: {hints}")

        if self.opening_hint:
            lines.append(f"OPENINGS: {self.opening_hint}")

        return "\n".join(lines)


class StructuralRAG:
    """Retrieves structural patterns from author corpus."""

    def __init__(self, author: str):
        self.author = author
        self.analyzer = get_structural_analyzer()
        self.indexer = get_indexer()
        self._cached_rhythms: List[RhythmFingerprint] = []
        self._loaded = False

    def load_patterns(self, sample_size: int = 50) -> int:
        """Load rhythm patterns from author's indexed corpus."""
        if self._loaded:
            return len(self._cached_rhythms)

        # Get chunks from indexer
        try:
            chunks = self.indexer.get_random_chunks(self.author, n=sample_size)
        except Exception as e:
            logger.warning(f"Could not load chunks for {self.author}: {e}")
            return 0

        # Analyze each chunk for rhythm
        for chunk_text in chunks:
            try:
                rhythm = self.analyzer.extract_rhythm(chunk_text)
                if rhythm.sentence_count >= 2:  # Need at least 2 sentences
                    self._cached_rhythms.append(rhythm)
            except Exception as e:
                logger.debug(f"Could not analyze chunk: {e}")
                continue

        self._loaded = True
        logger.info(f"Loaded {len(self._cached_rhythms)} rhythm patterns for {self.author}")
        return len(self._cached_rhythms)

    def get_rhythm_pattern(self, target_sentences: int = 4) -> str:
        """Get a rhythm pattern matching target sentence count."""
        if not self._cached_rhythms:
            self.load_patterns()

        if not self._cached_rhythms:
            # Fallback: generate Lovecraft-style pattern
            return self._generate_lovecraft_rhythm(target_sentences)

        # Find patterns with similar sentence count
        matching = [r for r in self._cached_rhythms
                   if abs(r.sentence_count - target_sentences) <= 2]

        if not matching:
            matching = self._cached_rhythms

        # Pick random matching pattern
        rhythm = random.choice(matching)
        return rhythm.to_rhythm_string()

    def _generate_lovecraft_rhythm(self, n: int) -> str:
        """Generate Lovecraft-style rhythm pattern."""
        # Lovecraft's characteristic patterns:
        # - Long complex sentences followed by short punchy ones
        # - Occasional fragments for emphasis
        # - Building tension with increasing length
        patterns = [
            ["LONG", "SHORT", "MEDIUM", "FRAGMENT", "LONG"],
            ["MEDIUM", "LONG", "SHORT", "LONG", "FRAGMENT"],
            ["SHORT", "LONG", "MEDIUM", "SHORT", "VERY_LONG"],
            ["LONG", "FRAGMENT", "LONG", "SHORT", "MEDIUM"],
            ["MEDIUM", "SHORT", "LONG", "FRAGMENT", "LONG"],
        ]
        pattern = random.choice(patterns)[:n]
        return " → ".join(pattern)

    def get_punctuation_hints(self) -> List[str]:
        """Get punctuation usage hints from corpus analysis."""
        if not self._cached_rhythms:
            self.load_patterns()

        hints = []

        if not self._cached_rhythms:
            # Lovecraft defaults
            return [
                "use em-dashes (—) for interruptions and asides",
                "use semicolons to link related clauses",
                "occasional ellipsis (...) for trailing thoughts",
            ]

        # Analyze cached rhythms for punctuation patterns
        dash_count = sum(1 for r in self._cached_rhythms if r.punctuation_density > 0.3)
        if dash_count > len(self._cached_rhythms) * 0.3:
            hints.append("use em-dashes (—) for interruptions and asides")

        fragment_ratio = sum(r.fragment_ratio for r in self._cached_rhythms) / len(self._cached_rhythms)
        if fragment_ratio > 0.1:
            hints.append(f"use sentence fragments (~{int(fragment_ratio*100)}% of sentences)")

        return hints if hints else ["vary punctuation for rhythm"]

    def get_length_guidance(self) -> str:
        """Get sentence length guidance from corpus."""
        if not self._cached_rhythms:
            self.load_patterns()

        if not self._cached_rhythms:
            return "Vary between 5 and 40 words per sentence"

        # Calculate range from corpus
        all_avgs = [r.avg_sentence_length for r in self._cached_rhythms]
        all_vars = [r.length_variance for r in self._cached_rhythms]

        avg_len = sum(all_avgs) / len(all_avgs)
        avg_var = sum(all_vars) / len(all_vars)

        min_len = max(3, int(avg_len - avg_var**0.5))
        max_len = int(avg_len + avg_var**0.5)

        if avg_var > 100:
            return f"HIGH variation: {min_len}-{max_len} words, mix short punchy with long flowing"
        elif avg_var > 50:
            return f"MODERATE variation: {min_len}-{max_len} words per sentence"
        else:
            return f"Sentences around {int(avg_len)} words, some variation"

    def get_fragment_hint(self) -> str:
        """Get fragment usage hint."""
        if not self._cached_rhythms:
            self.load_patterns()

        if not self._cached_rhythms:
            return "Include 1-2 sentence fragments for emphasis"

        avg_ratio = sum(r.fragment_ratio for r in self._cached_rhythms) / len(self._cached_rhythms)

        if avg_ratio > 0.2:
            return f"Use fragments liberally (~{int(avg_ratio*100)}% of sentences)"
        elif avg_ratio > 0.1:
            return "Include occasional fragments for punch"
        else:
            return "Use complete sentences primarily"

    def get_guidance(self, input_text: str) -> StructuralGuidance:
        """Get complete structural guidance for input text."""
        # Analyze input to match sentence count
        input_rhythm = self.analyzer.extract_rhythm(input_text)
        target_sentences = max(3, input_rhythm.sentence_count)

        return StructuralGuidance(
            rhythm_pattern=self.get_rhythm_pattern(target_sentences),
            punctuation_hints=self.get_punctuation_hints(),
            length_guidance=self.get_length_guidance(),
            fragment_hint=self.get_fragment_hint(),
            opening_hint="",  # Could add POS-based opening hints
        )


# Cache for structural RAG instances
_rag_cache: Dict[str, StructuralRAG] = {}


def get_structural_rag(author: str) -> StructuralRAG:
    """Get or create structural RAG for author."""
    if author not in _rag_cache:
        _rag_cache[author] = StructuralRAG(author)
    return _rag_cache[author]


def get_structural_guidance(author: str, input_text: str) -> str:
    """Convenience function to get formatted structural guidance."""
    rag = get_structural_rag(author)
    guidance = rag.get_guidance(input_text)
    return guidance.format_for_prompt()
