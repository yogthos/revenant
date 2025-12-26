"""Unit tests for rhythm planning."""

import pytest
from unittest.mock import MagicMock

from src.planning.rhythm_planner import RhythmPlanner, RhythmPattern
from src.models.style import AuthorProfile, StyleProfile
from src.models.graph import SemanticGraph, ParagraphRole, RhetoricalIntent


class TestRhythmPattern:
    """Test RhythmPattern dataclass."""

    def test_create_rhythm_pattern(self):
        """Test creating a rhythm pattern."""
        pattern = RhythmPattern(
            lengths=[10, 15, 8, 12],
            burstiness=0.25,
            avg_length=11.25
        )

        assert pattern.lengths == [10, 15, 8, 12]
        assert pattern.burstiness == 0.25
        assert pattern.avg_length == 11.25

    def test_len(self):
        """Test __len__ method."""
        pattern = RhythmPattern(lengths=[10, 15, 8], burstiness=0.2, avg_length=11.0)
        assert len(pattern) == 3

    def test_iter(self):
        """Test __iter__ method."""
        pattern = RhythmPattern(lengths=[10, 15, 8], burstiness=0.2, avg_length=11.0)
        assert list(pattern) == [10, 15, 8]

    def test_empty_pattern(self):
        """Test empty rhythm pattern."""
        pattern = RhythmPattern(lengths=[], burstiness=0.0, avg_length=0.0)
        assert len(pattern) == 0
        assert list(pattern) == []


class TestRhythmPlanner:
    """Test RhythmPlanner functionality."""

    @pytest.fixture
    def low_burstiness_profile(self):
        """Create a style profile with low burstiness."""
        author = AuthorProfile(
            name="Uniform Writer",
            style_dna="Consistent, measured prose.",
            avg_sentence_length=15.0,
            burstiness=0.1
        )
        return StyleProfile.from_author(author)

    @pytest.fixture
    def high_burstiness_profile(self):
        """Create a style profile with high burstiness."""
        author = AuthorProfile(
            name="Varied Writer",
            style_dna="Dynamic, varied prose.",
            avg_sentence_length=18.0,
            burstiness=0.4
        )
        return StyleProfile.from_author(author)

    @pytest.fixture
    def medium_burstiness_profile(self):
        """Create a style profile with medium burstiness."""
        author = AuthorProfile(
            name="Balanced Writer",
            style_dna="Balanced prose style.",
            avg_sentence_length=16.0,
            burstiness=0.25
        )
        return StyleProfile.from_author(author)

    def test_plan_rhythm_zero_sentences(self, low_burstiness_profile):
        """Test rhythm planning with zero sentences."""
        planner = RhythmPlanner(low_burstiness_profile)
        rhythm = planner.plan_rhythm(0)

        assert rhythm.lengths == []
        assert rhythm.burstiness == 0.0
        assert rhythm.avg_length == 0.0

    def test_plan_rhythm_single_sentence(self, low_burstiness_profile):
        """Test rhythm planning with single sentence."""
        planner = RhythmPlanner(low_burstiness_profile)
        rhythm = planner.plan_rhythm(1)

        assert len(rhythm.lengths) == 1
        assert rhythm.lengths[0] == 15  # avg_sentence_length
        assert rhythm.burstiness == 0.0

    def test_plan_rhythm_low_burstiness(self, low_burstiness_profile):
        """Test rhythm planning with low burstiness uses uniform pattern."""
        planner = RhythmPlanner(low_burstiness_profile)
        rhythm = planner.plan_rhythm(4)

        assert len(rhythm.lengths) == 4
        # Low burstiness should produce relatively uniform lengths
        avg = sum(rhythm.lengths) / len(rhythm.lengths)
        variance = sum((l - avg) ** 2 for l in rhythm.lengths) / len(rhythm.lengths)
        # Low variance expected
        assert variance < 50  # Relatively uniform

    def test_plan_rhythm_high_burstiness(self, high_burstiness_profile):
        """Test rhythm planning with high burstiness uses varied pattern."""
        planner = RhythmPlanner(high_burstiness_profile)
        rhythm = planner.plan_rhythm(4)

        assert len(rhythm.lengths) == 4
        # All lengths should be at least 5 (minimum)
        assert all(l >= 5 for l in rhythm.lengths)

    def test_plan_rhythm_medium_burstiness(self, medium_burstiness_profile):
        """Test rhythm planning with medium burstiness."""
        planner = RhythmPlanner(medium_burstiness_profile)
        rhythm = planner.plan_rhythm(4)

        assert len(rhythm.lengths) == 4
        # Should be around the target average
        avg = sum(rhythm.lengths) / len(rhythm.lengths)
        assert 10 <= avg <= 25

    def test_plan_for_propositions(self, low_burstiness_profile):
        """Test planning based on proposition count."""
        planner = RhythmPlanner(low_burstiness_profile)
        rhythm = planner.plan_for_propositions(6)

        # 6 propositions / 1.5 props per sentence ≈ 4 sentences
        assert 2 <= len(rhythm.lengths) <= 6

    def test_plan_for_propositions_intro_role(self, low_burstiness_profile):
        """Test planning adjusts for intro paragraphs."""
        planner = RhythmPlanner(low_burstiness_profile)

        graph = SemanticGraph(
            nodes=[],
            edges=[],
            role=ParagraphRole.INTRO
        )

        # Intro paragraphs get fewer, longer sentences
        rhythm = planner.plan_for_propositions(6, graph)
        assert len(rhythm.lengths) >= 1

    def test_plan_for_propositions_conclusion_role(self, low_burstiness_profile):
        """Test planning adjusts for conclusion paragraphs."""
        planner = RhythmPlanner(low_burstiness_profile)

        graph = SemanticGraph(
            nodes=[],
            edges=[],
            role=ParagraphRole.CONCLUSION
        )

        # Conclusions are concise (max 3 sentences)
        rhythm = planner.plan_for_propositions(10, graph)
        assert len(rhythm.lengths) <= 3

    def test_minimum_sentence_length(self, high_burstiness_profile):
        """Test that sentence lengths don't go below minimum."""
        planner = RhythmPlanner(high_burstiness_profile)

        # Run multiple times to check randomness doesn't violate minimum
        for _ in range(5):
            rhythm = planner.plan_rhythm(5)
            assert all(l >= 5 for l in rhythm.lengths)

    def test_get_extended_pattern_truncate(self, low_burstiness_profile):
        """Test pattern extension when truncating."""
        planner = RhythmPlanner(low_burstiness_profile)

        # Get pattern shorter than base
        pattern = planner._get_extended_pattern("varied", 2)
        assert len(pattern) == 2

    def test_get_extended_pattern_extend(self, low_burstiness_profile):
        """Test pattern extension when extending."""
        planner = RhythmPlanner(low_burstiness_profile)

        # Get pattern longer than base (base has 4 elements)
        pattern = planner._get_extended_pattern("uniform", 6)
        assert len(pattern) == 6

    def test_scale_pattern(self, low_burstiness_profile):
        """Test scaling pattern to target average."""
        planner = RhythmPlanner(low_burstiness_profile)

        pattern = [1.0, 1.0, 1.0, 1.0]
        lengths = planner._scale_pattern(pattern, 15.0)

        # All should be around 15
        assert all(12 <= l <= 18 for l in lengths)

    def test_add_variation_low_burstiness(self, low_burstiness_profile):
        """Test variation addition with low burstiness."""
        planner = RhythmPlanner(low_burstiness_profile)

        lengths = [15, 15, 15, 15]
        varied = planner._add_variation(lengths, 0.05)

        # Should be unchanged with very low burstiness
        assert varied == lengths

    def test_calculate_burstiness_uniform(self, low_burstiness_profile):
        """Test burstiness calculation with uniform lengths."""
        planner = RhythmPlanner(low_burstiness_profile)

        # Perfectly uniform = 0 burstiness
        burstiness = planner._calculate_burstiness([10, 10, 10, 10])
        assert burstiness == 0.0

    def test_calculate_burstiness_varied(self, low_burstiness_profile):
        """Test burstiness calculation with varied lengths."""
        planner = RhythmPlanner(low_burstiness_profile)

        # Varied lengths = positive burstiness
        burstiness = planner._calculate_burstiness([5, 20, 8, 15])
        assert burstiness > 0.0

    def test_calculate_burstiness_single(self, low_burstiness_profile):
        """Test burstiness calculation with single length."""
        planner = RhythmPlanner(low_burstiness_profile)

        burstiness = planner._calculate_burstiness([15])
        assert burstiness == 0.0

    def test_calculate_burstiness_empty(self, low_burstiness_profile):
        """Test burstiness calculation with empty list."""
        planner = RhythmPlanner(low_burstiness_profile)

        burstiness = planner._calculate_burstiness([])
        assert burstiness == 0.0

    def test_achieved_burstiness_near_target(self, medium_burstiness_profile):
        """Test that achieved burstiness is somewhat near target."""
        planner = RhythmPlanner(medium_burstiness_profile)

        # Run multiple times to check average
        burstiness_values = []
        for _ in range(10):
            rhythm = planner.plan_rhythm(5)
            burstiness_values.append(rhythm.burstiness)

        avg_burstiness = sum(burstiness_values) / len(burstiness_values)
        # Should be in reasonable range
        assert 0.0 <= avg_burstiness <= 0.6
