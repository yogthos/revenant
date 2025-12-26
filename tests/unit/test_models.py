"""Unit tests for data models."""

import pytest
from src.models import (
    Message,
    MessageRole,
    LLMResponse,
    ValidationResult,
    InputIssue,
    AuthorProfile,
    StyleProfile,
)


class TestMessage:
    """Test Message model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_to_dict(self):
        """Test converting message to dict."""
        msg = Message(role=MessageRole.SYSTEM, content="You are helpful")
        d = msg.to_dict()
        assert d["role"] == "system"
        assert d["content"] == "You are helpful"

    def test_all_roles(self):
        """Test all message roles."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello world",
            input_tokens=10,
            output_tokens=5,
            model="test-model"
        )
        assert response.content == "Hello world"
        assert response.input_tokens == 10
        assert response.output_tokens == 5
        assert response.model == "test-model"

    def test_total_tokens(self):
        """Test total tokens calculation."""
        response = LLMResponse(
            content="test",
            input_tokens=100,
            output_tokens=50
        )
        assert response.total_tokens == 150

    def test_default_values(self):
        """Test default values."""
        response = LLMResponse(content="test")
        assert response.input_tokens == 0
        assert response.output_tokens == 0
        assert response.model == ""


class TestValidationResult:
    """Test ValidationResult model."""

    def test_valid_result(self):
        """Test valid validation result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.issues == []
        assert result.recommendation is None

    def test_invalid_result(self):
        """Test invalid validation result with issues."""
        result = ValidationResult(
            valid=False,
            issues=["Too short", "Missing context"],
            recommendation="Add more content"
        )
        assert result.valid is False
        assert len(result.issues) == 2
        assert result.recommendation == "Add more content"


class TestInputIssue:
    """Test InputIssue enum."""

    def test_issue_values(self):
        """Test all input issue types."""
        assert InputIssue.TOO_SHORT.value == "too_short"
        assert InputIssue.ALREADY_IN_STYLE.value == "already_in_style"
        assert InputIssue.LIST_ONLY.value == "list_only"
        assert InputIssue.MALFORMED.value == "malformed"


class TestAuthorProfile:
    """Test AuthorProfile model."""

    def test_create_profile(self):
        """Test creating an author profile."""
        profile = AuthorProfile(
            name="Test Author",
            style_dna="Writes with clarity and precision",
            top_vocab=["however", "therefore", "indeed"],
            avg_sentence_length=18.5,
            burstiness=0.4,
            punctuation_freq={"em_dash": 0.05, "semicolon": 0.02},
            perspective="first_person_singular"
        )
        assert profile.name == "Test Author"
        assert profile.style_dna == "Writes with clarity and precision"
        assert len(profile.top_vocab) == 3
        assert profile.avg_sentence_length == 18.5
        assert profile.burstiness == 0.4
        assert profile.perspective == "first_person_singular"

    def test_default_values(self):
        """Test default values for author profile."""
        profile = AuthorProfile(name="Author", style_dna="DNA")
        assert profile.top_vocab == []
        assert profile.avg_sentence_length == 15.0
        assert profile.burstiness == 0.3
        assert profile.punctuation_freq == {}
        assert profile.perspective == "third_person"

    def test_to_dict(self):
        """Test converting author profile to dict."""
        profile = AuthorProfile(
            name="Test",
            style_dna="DNA",
            top_vocab=["word1", "word2"],
            avg_sentence_length=20.0,
            burstiness=0.5,
            punctuation_freq={"comma": 0.1},
            perspective="third_person"
        )
        d = profile.to_dict()
        assert d["name"] == "Test"
        assert d["style_dna"] == "DNA"
        assert d["top_vocab"] == ["word1", "word2"]
        assert d["avg_sentence_length"] == 20.0

    def test_from_dict(self):
        """Test creating author profile from dict."""
        data = {
            "name": "From Dict Author",
            "style_dna": "From dict DNA",
            "top_vocab": ["a", "b", "c"],
            "avg_sentence_length": 12.0,
            "burstiness": 0.2,
            "punctuation_freq": {"period": 0.5},
            "perspective": "first_person_plural"
        }
        profile = AuthorProfile.from_dict(data)
        assert profile.name == "From Dict Author"
        assert profile.style_dna == "From dict DNA"
        assert profile.avg_sentence_length == 12.0
        assert profile.perspective == "first_person_plural"

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        data = {"name": "Minimal"}
        profile = AuthorProfile.from_dict(data)
        assert profile.name == "Minimal"
        assert profile.style_dna == ""
        assert profile.avg_sentence_length == 15.0


class TestStyleProfile:
    """Test StyleProfile model."""

    def test_single_author_profile(self):
        """Test single author style profile."""
        author = AuthorProfile(
            name="Single Author",
            style_dna="Clear and concise",
            top_vocab=["clear", "concise"],
            avg_sentence_length=15.0,
            burstiness=0.3
        )
        profile = StyleProfile(primary_author=author)

        assert profile.get_author_name() == "Single Author"
        assert profile.get_effective_style_dna() == "Clear and concise"
        assert profile.get_effective_vocab() == ["clear", "concise"]
        assert profile.get_effective_avg_sentence_length() == 15.0
        assert profile.get_effective_burstiness() == 0.3

    def test_from_author_factory(self):
        """Test creating style profile from single author."""
        author = AuthorProfile(name="Factory Author", style_dna="DNA")
        profile = StyleProfile.from_author(author)

        assert profile.primary_author == author
        assert profile.secondary_author is None
        assert profile.blend_ratio == 1.0

    def test_blended_profile_sentence_length(self):
        """Test blended profile calculates weighted sentence length."""
        author1 = AuthorProfile(name="A1", style_dna="D1", avg_sentence_length=10.0)
        author2 = AuthorProfile(name="A2", style_dna="D2", avg_sentence_length=20.0)

        # 50/50 blend
        profile = StyleProfile(
            primary_author=author1,
            secondary_author=author2,
            blend_ratio=0.5
        )
        # Expected: 0.5 * 10 + 0.5 * 20 = 15
        assert profile.get_effective_avg_sentence_length() == 15.0

    def test_blended_profile_burstiness(self):
        """Test blended profile calculates weighted burstiness."""
        author1 = AuthorProfile(name="A1", style_dna="D1", burstiness=0.2)
        author2 = AuthorProfile(name="A2", style_dna="D2", burstiness=0.6)

        # 75/25 blend
        profile = StyleProfile(
            primary_author=author1,
            secondary_author=author2,
            blend_ratio=0.75
        )
        # Expected: 0.75 * 0.2 + 0.25 * 0.6 = 0.15 + 0.15 = 0.3
        assert profile.get_effective_burstiness() == pytest.approx(0.3)

    def test_full_primary_blend_ratio(self):
        """Test that blend_ratio=1.0 uses only primary."""
        author1 = AuthorProfile(name="A1", style_dna="D1", avg_sentence_length=10.0)
        author2 = AuthorProfile(name="A2", style_dna="D2", avg_sentence_length=20.0)

        profile = StyleProfile(
            primary_author=author1,
            secondary_author=author2,
            blend_ratio=1.0
        )
        assert profile.get_effective_avg_sentence_length() == 10.0

    def test_no_secondary_uses_primary(self):
        """Test that missing secondary uses primary regardless of ratio."""
        author1 = AuthorProfile(name="A1", style_dna="D1", avg_sentence_length=10.0)

        profile = StyleProfile(
            primary_author=author1,
            secondary_author=None,
            blend_ratio=0.5
        )
        # Should still use primary since secondary is None
        assert profile.get_effective_avg_sentence_length() == 10.0
