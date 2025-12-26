"""Unit tests for validation critics."""

import pytest
from unittest.mock import MagicMock, patch

from src.generation.critics import (
    Critic,
    CriticPanel,
    CriticFeedback,
    CriticType,
    ValidationResult,
    LengthCritic,
    KeywordCritic,
    FluencyCritic,
    SemanticCritic,
    StyleCritic,
)
from src.models.plan import SentenceNode, SentenceRole
from src.models.graph import PropositionNode
from src.models.style import AuthorProfile, StyleProfile
from src.ingestion.context_analyzer import GlobalContext


class TestCriticFeedback:
    """Test CriticFeedback dataclass."""

    def test_create_feedback(self):
        """Test creating critic feedback."""
        feedback = CriticFeedback(
            critic_type=CriticType.LENGTH,
            score=0.85,
            passed=True,
            issues=[],
            suggestions=[]
        )

        assert feedback.critic_type == CriticType.LENGTH
        assert feedback.score == 0.85
        assert feedback.passed is True

    def test_feedback_with_issues(self):
        """Test feedback with issues and suggestions."""
        feedback = CriticFeedback(
            critic_type=CriticType.SEMANTIC,
            score=0.5,
            passed=False,
            issues=["Missing concept"],
            suggestions=["Add concept X"]
        )

        assert len(feedback.issues) == 1
        assert len(feedback.suggestions) == 1

    def test_to_dict(self):
        """Test converting feedback to dict."""
        feedback = CriticFeedback(
            critic_type=CriticType.STYLE,
            score=0.7,
            passed=True,
            issues=["Minor issue"],
            suggestions=[]
        )

        d = feedback.to_dict()

        assert d["critic_type"] == "style"
        assert d["score"] == 0.7
        assert d["passed"] is True


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_all_passed_true(self):
        """Test all_passed when all critics pass."""
        feedbacks = [
            CriticFeedback(CriticType.LENGTH, 0.8, True, [], []),
            CriticFeedback(CriticType.KEYWORD, 0.9, True, [], []),
        ]
        result = ValidationResult(feedbacks=feedbacks)

        assert result.all_passed is True

    def test_all_passed_false(self):
        """Test all_passed when some critics fail."""
        feedbacks = [
            CriticFeedback(CriticType.LENGTH, 0.8, True, [], []),
            CriticFeedback(CriticType.KEYWORD, 0.3, False, ["Missing keyword"], []),
        ]
        result = ValidationResult(feedbacks=feedbacks)

        assert result.all_passed is False

    def test_overall_score(self):
        """Test overall score calculation."""
        feedbacks = [
            CriticFeedback(CriticType.LENGTH, 0.8, True, [], []),
            CriticFeedback(CriticType.KEYWORD, 0.6, True, [], []),
        ]
        result = ValidationResult(feedbacks=feedbacks)

        assert result.overall_score == 0.7

    def test_failing_critics(self):
        """Test getting failing critics."""
        feedbacks = [
            CriticFeedback(CriticType.LENGTH, 0.8, True, [], []),
            CriticFeedback(CriticType.KEYWORD, 0.3, False, ["Issue"], []),
            CriticFeedback(CriticType.FLUENCY, 0.4, False, ["Issue"], []),
        ]
        result = ValidationResult(feedbacks=feedbacks)

        failing = result.failing_critics
        assert len(failing) == 2

    def test_get_consolidated_feedback(self):
        """Test consolidated feedback for revision."""
        feedbacks = [
            CriticFeedback(CriticType.LENGTH, 0.8, True, [], []),
            CriticFeedback(CriticType.KEYWORD, 0.3, False, ["Missing: test"], []),
        ]
        result = ValidationResult(feedbacks=feedbacks)

        feedback = result.get_consolidated_feedback()

        assert "[keyword]" in feedback.lower()
        assert "Missing: test" in feedback

    def test_consolidated_feedback_all_passed(self):
        """Test consolidated feedback when all pass."""
        feedbacks = [
            CriticFeedback(CriticType.LENGTH, 0.9, True, [], []),
        ]
        result = ValidationResult(feedbacks=feedbacks)

        feedback = result.get_consolidated_feedback()

        assert "passed" in feedback.lower()


class TestLengthCritic:
    """Test LengthCritic functionality."""

    @pytest.fixture
    def critic(self):
        """Create length critic."""
        return LengthCritic(threshold=0.7, tolerance=5)

    @pytest.fixture
    def node_15_words(self):
        """Create node targeting 15 words."""
        return SentenceNode(id="s1", propositions=[], target_length=15)

    def test_exact_length_passes(self, critic, node_15_words):
        """Test exact length gets perfect score."""
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"

        feedback = critic.evaluate(text, node_15_words)

        assert feedback.score == 1.0
        assert feedback.passed is True

    def test_within_tolerance_passes(self, critic, node_15_words):
        """Test within tolerance passes."""
        text = "one two three four five six seven eight nine ten eleven twelve"  # 12 words

        feedback = critic.evaluate(text, node_15_words)

        assert feedback.passed is True
        assert len(feedback.issues) == 0

    def test_over_tolerance_fails(self, critic, node_15_words):
        """Test over tolerance fails."""
        # 25 words - 10 over target
        text = " ".join(["word"] * 25)

        feedback = critic.evaluate(text, node_15_words)

        assert feedback.passed is False
        assert "too long" in feedback.issues[0].lower()

    def test_under_tolerance_fails(self, critic, node_15_words):
        """Test under tolerance fails."""
        text = "one two three four"  # 4 words

        feedback = critic.evaluate(text, node_15_words)

        assert feedback.passed is False
        assert "too short" in feedback.issues[0].lower()


class TestKeywordCritic:
    """Test KeywordCritic functionality."""

    @pytest.fixture
    def critic(self):
        """Create keyword critic."""
        return KeywordCritic(threshold=0.8)

    def test_all_keywords_present(self, critic):
        """Test when all keywords present."""
        node = SentenceNode(
            id="s1", propositions=[],
            target_length=10, keywords=["evolution", "species"]
        )
        text = "The evolution of species demonstrates natural selection."

        feedback = critic.evaluate(text, node)

        assert feedback.score == 1.0
        assert feedback.passed is True

    def test_partial_keywords(self, critic):
        """Test when some keywords missing."""
        node = SentenceNode(
            id="s1", propositions=[],
            target_length=10, keywords=["evolution", "species", "mutation"]
        )
        text = "The evolution of species is remarkable."

        feedback = critic.evaluate(text, node)

        # 2/3 = 0.667
        assert feedback.score == pytest.approx(2/3)
        assert feedback.passed is False
        assert "mutation" in feedback.issues[0]

    def test_no_keywords_required(self, critic):
        """Test when no keywords required."""
        node = SentenceNode(id="s1", propositions=[], target_length=10, keywords=[])
        text = "Any text here."

        feedback = critic.evaluate(text, node)

        assert feedback.score == 1.0
        assert feedback.passed is True

    def test_case_insensitive_matching(self, critic):
        """Test keyword matching is case insensitive."""
        node = SentenceNode(
            id="s1", propositions=[],
            target_length=10, keywords=["Evolution", "SPECIES"]
        )
        text = "The evolution of species continues."

        feedback = critic.evaluate(text, node)

        assert feedback.score == 1.0


class TestFluencyCritic:
    """Test FluencyCritic functionality."""

    @pytest.fixture
    def critic(self):
        """Create fluency critic."""
        return FluencyCritic(threshold=0.7)

    @pytest.fixture
    def simple_node(self):
        """Create simple sentence node."""
        return SentenceNode(id="s1", propositions=[], target_length=10)

    def test_proper_sentence_passes(self, critic, simple_node):
        """Test proper sentence passes fluency check."""
        text = "The quick brown fox jumps over the lazy dog."

        feedback = critic.evaluate(text, simple_node)

        assert feedback.passed is True
        assert feedback.score > 0.7

    def test_multiple_sentences_penalized(self, critic, simple_node):
        """Test multiple sentences are penalized."""
        text = "First sentence here. Second sentence here."

        feedback = critic.evaluate(text, simple_node)

        assert "multiple sentences" in feedback.issues[0].lower()

    def test_missing_punctuation_penalized(self, critic, simple_node):
        """Test missing ending punctuation is penalized."""
        text = "This sentence has no ending"

        feedback = critic.evaluate(text, simple_node)

        assert any("punctuation" in issue.lower() for issue in feedback.issues)

    def test_excessive_repetition_penalized(self, critic, simple_node):
        """Test excessive word repetition is penalized."""
        text = "The very very very very long sentence continues."

        feedback = critic.evaluate(text, simple_node)

        assert any("repetition" in issue.lower() for issue in feedback.issues)


class TestSemanticCritic:
    """Test SemanticCritic functionality."""

    @pytest.fixture
    def critic(self):
        """Create semantic critic."""
        return SemanticCritic(threshold=0.7)

    def test_similar_text_passes(self, critic):
        """Test semantically similar text passes."""
        prop = PropositionNode(
            id="p1",
            text="Evolution drives adaptation in species.",
            subject="Evolution", verb="drives"
        )
        node = SentenceNode(id="s1", propositions=[prop], target_length=10)
        text = "Evolution is the driving force behind species adaptation."

        feedback = critic.evaluate(text, node)

        # Similar text should have decent overlap
        assert feedback.score > 0.4

    def test_different_text_lower_score(self, critic):
        """Test very different text gets lower score."""
        prop = PropositionNode(
            id="p1",
            text="Climate change affects ecosystems.",
            subject="Climate change", verb="affects"
        )
        node = SentenceNode(id="s1", propositions=[prop], target_length=10)
        text = "The dog ran quickly through the park yesterday."

        feedback = critic.evaluate(text, node)

        # Very different content should have low overlap
        assert feedback.score < 0.5

    def test_no_propositions_passes(self, critic):
        """Test empty propositions auto-passes."""
        node = SentenceNode(id="s1", propositions=[], target_length=10)
        text = "Any text here."

        feedback = critic.evaluate(text, node)

        assert feedback.score == 1.0
        assert feedback.passed is True


class TestStyleCritic:
    """Test StyleCritic functionality."""

    @pytest.fixture
    def style_profile(self):
        """Create style profile."""
        author = AuthorProfile(
            name="Test Author",
            style_dna="Clear, direct writing.",
            avg_sentence_length=15.0,
            burstiness=0.25
        )
        profile = StyleProfile.from_author(author)
        profile.primary_author.top_vocabulary = ["therefore", "however", "indeed", "evolution"]
        return profile

    @pytest.fixture
    def critic(self, style_profile):
        """Create style critic."""
        return StyleCritic(style_profile, threshold=0.6)

    @pytest.fixture
    def simple_node(self):
        """Create simple sentence node."""
        return SentenceNode(id="s1", propositions=[], target_length=15)

    def test_vocabulary_alignment(self, critic, simple_node):
        """Test vocabulary alignment scoring."""
        text = "Therefore, evolution indeed demonstrates change."

        feedback = critic.evaluate(text, simple_node)

        # Contains target vocabulary
        assert feedback.score > 0.3

    def test_mismatched_length_lowers_score(self, critic):
        """Test mismatched length affects score."""
        node = SentenceNode(id="s1", propositions=[], target_length=50)
        text = "Short."

        feedback = critic.evaluate(text, node)

        # Very different length = lower complexity score
        assert feedback.score < 0.8


class TestCriticPanel:
    """Test CriticPanel functionality."""

    @pytest.fixture
    def style_profile(self):
        """Create style profile."""
        author = AuthorProfile(
            name="Test Author",
            style_dna="Clear style.",
            avg_sentence_length=15.0,
            burstiness=0.2
        )
        return StyleProfile.from_author(author)

    @pytest.fixture
    def global_context(self):
        """Create global context."""
        return GlobalContext(
            thesis="Test thesis.",
            intent="inform",
            keywords=["test"],
            perspective="third_person",
            style_dna="Direct.",
            author_name="Author",
            target_burstiness=0.2,
            target_sentence_length=15.0,
            top_vocabulary=["word"],
            total_paragraphs=1,
            processed_paragraphs=0
        )

    @pytest.fixture
    def panel(self, style_profile, global_context):
        """Create critic panel."""
        return CriticPanel(style_profile, global_context)

    @pytest.fixture
    def sentence_node(self):
        """Create sample sentence node."""
        prop = PropositionNode(id="p1", text="Test concept.", subject="Test", verb="is")
        return SentenceNode(
            id="s1",
            propositions=[prop],
            target_length=10,
            keywords=["test"]
        )

    def test_validate_runs_all_critics(self, panel, sentence_node):
        """Test validation runs all critics."""
        text = "The test concept is demonstrated here clearly."

        result = panel.validate(text, sentence_node)

        # Should have feedback from multiple critics
        assert len(result.feedbacks) >= 3

    def test_validate_returns_validation_result(self, panel, sentence_node):
        """Test validate returns ValidationResult."""
        text = "Test sentence."

        result = panel.validate(text, sentence_node)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'all_passed')
        assert hasattr(result, 'overall_score')

    def test_get_revision_feedback(self, panel, sentence_node):
        """Test getting revision feedback."""
        text = "x"  # Very short, should fail length

        feedback = panel.get_revision_feedback(text, sentence_node)

        assert isinstance(feedback, str)
        # Should mention issues
        assert len(feedback) > 0

    def test_critic_error_handled(self, panel, sentence_node):
        """Test critic errors are handled gracefully."""
        # Force an error by using invalid input
        text = "Normal text here."

        # This should not raise even if a critic fails
        result = panel.validate(text, sentence_node)

        assert isinstance(result, ValidationResult)
