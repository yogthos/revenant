"""Rhythm planning for sentence length variation."""

import random
from dataclasses import dataclass
from typing import List, Optional

from ..models.style import StyleProfile
from ..models.graph import SemanticGraph
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RhythmPattern:
    """A rhythm pattern for sentence lengths."""
    lengths: List[int]  # Target lengths in words
    burstiness: float  # Achieved burstiness
    avg_length: float  # Average length

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        return iter(self.lengths)


class RhythmPlanner:
    """Plans sentence rhythm patterns based on author style.

    Creates sentence length targets that match the author's
    characteristic burstiness (variation in sentence lengths).
    """

    # Common rhythm patterns
    PATTERNS = {
        "uniform": [1.0, 1.0, 1.0, 1.0],  # All same length
        "building": [0.7, 0.9, 1.1, 1.3],  # Gradually longer
        "climactic": [0.8, 0.9, 1.4, 0.9],  # Build to peak, then down
        "varied": [1.2, 0.7, 1.1, 0.9],  # Irregular variation
        "bookend": [1.2, 0.8, 0.8, 1.2],  # Strong start and end
    }

    def __init__(self, style_profile: StyleProfile):
        """Initialize rhythm planner.

        Args:
            style_profile: Target style profile.
        """
        self.style_profile = style_profile
        self.target_avg = style_profile.get_effective_avg_sentence_length()
        self.target_burstiness = style_profile.get_effective_burstiness()

    def plan_rhythm(
        self,
        num_sentences: int,
        semantic_graph: Optional[SemanticGraph] = None
    ) -> RhythmPattern:
        """Plan rhythm pattern for a paragraph.

        Args:
            num_sentences: Number of sentences to plan.
            semantic_graph: Optional semantic graph for context.

        Returns:
            RhythmPattern with target lengths.
        """
        if num_sentences <= 0:
            return RhythmPattern(lengths=[], burstiness=0.0, avg_length=0.0)

        if num_sentences == 1:
            return RhythmPattern(
                lengths=[int(self.target_avg)],
                burstiness=0.0,
                avg_length=self.target_avg
            )

        # Select base pattern based on burstiness
        if self.target_burstiness < 0.15:
            pattern_name = "uniform"
        elif self.target_burstiness < 0.3:
            pattern_name = random.choice(["building", "bookend"])
        else:
            pattern_name = random.choice(["varied", "climactic"])

        # Get base pattern and extend/truncate to match num_sentences
        base_pattern = self._get_extended_pattern(pattern_name, num_sentences)

        # Scale to target average length
        lengths = self._scale_pattern(base_pattern, self.target_avg)

        # Add controlled randomness based on burstiness
        lengths = self._add_variation(lengths, self.target_burstiness)

        # Ensure minimum lengths
        lengths = [max(5, l) for l in lengths]

        # Calculate achieved statistics
        achieved_avg = sum(lengths) / len(lengths)
        achieved_burstiness = self._calculate_burstiness(lengths)

        return RhythmPattern(
            lengths=lengths,
            burstiness=achieved_burstiness,
            avg_length=achieved_avg
        )

    def plan_for_propositions(
        self,
        num_propositions: int,
        semantic_graph: Optional[SemanticGraph] = None
    ) -> RhythmPattern:
        """Plan rhythm based on number of propositions.

        Args:
            num_propositions: Number of propositions to express.
            semantic_graph: Optional semantic graph.

        Returns:
            RhythmPattern with appropriate number of sentences.
        """
        # Estimate propositions per sentence (typically 1-2)
        props_per_sentence = 1.5
        num_sentences = max(1, int(num_propositions / props_per_sentence + 0.5))

        # Adjust for paragraph role if available
        if semantic_graph and semantic_graph.role:
            role = semantic_graph.role.value
            if role == "INTRO":
                # Introductions often have slightly fewer, longer sentences
                num_sentences = max(1, num_sentences - 1)
            elif role == "CONCLUSION":
                # Conclusions are often concise
                num_sentences = max(1, min(num_sentences, 3))

        return self.plan_rhythm(num_sentences, semantic_graph)

    def _get_extended_pattern(self, pattern_name: str, length: int) -> List[float]:
        """Get pattern extended or truncated to target length.

        Args:
            pattern_name: Name of base pattern.
            length: Target length.

        Returns:
            Pattern list of multipliers.
        """
        base = self.PATTERNS.get(pattern_name, self.PATTERNS["varied"])

        if len(base) == length:
            return base.copy()

        if len(base) > length:
            # Truncate, keeping first and last
            if length == 1:
                return [base[0]]
            elif length == 2:
                return [base[0], base[-1]]
            else:
                # Keep first, last, and sample middle
                middle_indices = list(range(1, len(base) - 1))
                sample_size = length - 2
                if sample_size > 0:
                    sampled = sorted(random.sample(
                        middle_indices,
                        min(sample_size, len(middle_indices))
                    ))
                    return [base[0]] + [base[i] for i in sampled] + [base[-1]]
                return [base[0], base[-1]]

        # Extend by interpolation/repetition
        extended = []
        for i in range(length):
            # Map index to base pattern
            base_idx = int(i * len(base) / length)
            extended.append(base[base_idx])

        return extended

    def _scale_pattern(self, pattern: List[float], target_avg: float) -> List[int]:
        """Scale pattern to target average length.

        Args:
            pattern: Pattern of multipliers.
            target_avg: Target average length.

        Returns:
            List of integer lengths.
        """
        # Normalize pattern to sum to len(pattern)
        pattern_sum = sum(pattern)
        normalized = [p * len(pattern) / pattern_sum for p in pattern]

        # Scale to target average
        lengths = [int(n * target_avg) for n in normalized]

        return lengths

    def _add_variation(self, lengths: List[int], burstiness: float) -> List[int]:
        """Add random variation based on burstiness.

        Args:
            lengths: Base lengths.
            burstiness: Target burstiness.

        Returns:
            Lengths with added variation.
        """
        if burstiness < 0.1 or len(lengths) < 2:
            return lengths

        # Calculate variation amount
        avg = sum(lengths) / len(lengths)
        variation = avg * burstiness * 0.5  # Scale factor

        varied = []
        for length in lengths:
            delta = random.gauss(0, variation)
            varied.append(max(5, int(length + delta)))

        return varied

    def _calculate_burstiness(self, lengths: List[int]) -> float:
        """Calculate burstiness of length sequence.

        Args:
            lengths: Sentence lengths.

        Returns:
            Burstiness (coefficient of variation).
        """
        if len(lengths) < 2:
            return 0.0

        avg = sum(lengths) / len(lengths)
        if avg == 0:
            return 0.0

        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5

        return std_dev / avg
