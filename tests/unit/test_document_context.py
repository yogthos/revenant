"""Tests for the document context module."""

import pytest
from unittest.mock import MagicMock

from src.generation.document_context import (
    DocumentContext,
    DocumentContextExtractor,
    extract_document_context,
)


class TestDocumentContext:
    """Tests for DocumentContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = DocumentContext()
        assert ctx.thesis == ""
        assert ctx.intent == ""
        assert ctx.key_entities == []
        assert ctx.key_concepts == []
        assert ctx.tone == ""
        assert ctx.total_paragraphs == 0
        assert ctx.total_words == 0

    def test_to_critic_context_empty(self):
        """Test critic context with empty values."""
        ctx = DocumentContext()
        result = ctx.to_critic_context()
        assert result == ""

    def test_to_critic_context_full(self):
        """Test critic context with full values."""
        ctx = DocumentContext(
            thesis="Evolution explains biodiversity",
            intent="informative",
            key_entities=["Darwin", "Galapagos", "finches"],
            tone="academic",
        )
        result = ctx.to_critic_context()
        assert "Evolution explains biodiversity" in result
        assert "informative" in result
        assert "Darwin" in result
        assert "academic" in result

    def test_to_generation_hint_empty(self):
        """Test generation hint with empty values."""
        ctx = DocumentContext()
        result = ctx.to_generation_hint()
        assert result == ""

    def test_to_generation_hint_with_intent_and_tone(self):
        """Test generation hint with intent and tone."""
        ctx = DocumentContext(intent="persuasive", tone="formal")
        result = ctx.to_generation_hint()
        assert "formal" in result
        assert "persuasive" in result

    def test_to_generation_hint_with_intent_only(self):
        """Test generation hint with intent only."""
        ctx = DocumentContext(intent="narrative")
        result = ctx.to_generation_hint()
        assert result == "narrative text"

    def test_to_generation_hint_with_tone_only(self):
        """Test generation hint with tone only."""
        ctx = DocumentContext(tone="conversational")
        result = ctx.to_generation_hint()
        assert result == "conversational text"


class TestDocumentContextExtractor:
    """Tests for DocumentContextExtractor."""

    def test_basic_extraction_without_llm(self):
        """Test extraction using heuristics only."""
        text = """
        The theory of evolution by natural selection is a cornerstone of modern biology.
        Darwin observed that species adapt to their environments over time.

        This process occurs through random mutations and selective pressures.
        Species that are better adapted survive and reproduce more successfully.
        """
        extractor = DocumentContextExtractor()
        ctx = extractor.extract(text)

        assert ctx.total_paragraphs > 0
        assert ctx.total_words > 0
        assert len(ctx.key_concepts) > 0
        assert ctx.tone != ""
        assert ctx.intent != ""

    def test_extraction_with_academic_tone(self):
        """Test detection of academic tone."""
        text = """
        This research study examines the hypothesis that climate change
        affects migration patterns. Our findings suggest a correlation
        between temperature increase and behavioral changes.
        """
        extractor = DocumentContextExtractor()
        ctx = extractor.extract(text)
        assert ctx.tone == "academic"

    def test_extraction_with_conversational_tone(self):
        """Test detection of conversational tone."""
        text = """
        Let me tell you about something really cool I discovered.
        You won't believe what happened when we tried this experiment.
        I think you'll find this fascinating.
        """
        extractor = DocumentContextExtractor()
        ctx = extractor.extract(text)
        assert ctx.tone == "conversational"

    def test_extraction_with_formal_tone(self):
        """Test detection of formal tone."""
        text = """
        Furthermore, the matter indicates a significant change.
        Therefore, we must conclude that the initial position is correct.
        Consequently, additional measures are required. Hence the change.
        """
        extractor = DocumentContextExtractor()
        ctx = extractor.extract(text)
        assert ctx.tone == "formal"

    def test_extraction_with_mock_llm(self):
        """Test extraction with mock LLM provider."""
        mock_provider = MagicMock()
        mock_provider.call.return_value = '{"thesis": "Test thesis", "intent": "informative", "tone": "formal"}'

        text = "This is a test document with some content."
        extractor = DocumentContextExtractor(llm_provider=mock_provider)
        ctx = extractor.extract(text)

        assert ctx.thesis == "Test thesis"
        assert ctx.intent == "informative"
        assert ctx.tone == "formal"
        mock_provider.call.assert_called_once()

    def test_extraction_handles_llm_failure(self):
        """Test that extraction falls back to heuristics on LLM failure."""
        mock_provider = MagicMock()
        mock_provider.call.side_effect = Exception("API Error")

        text = """
        Furthermore, the research demonstrates significant findings.
        This study examines multiple factors affecting the outcome.
        """
        extractor = DocumentContextExtractor(llm_provider=mock_provider)
        ctx = extractor.extract(text)

        # Should still get results from heuristics
        assert ctx.tone != ""
        assert ctx.intent != ""

    def test_extraction_handles_invalid_json(self):
        """Test that extraction handles invalid JSON from LLM."""
        mock_provider = MagicMock()
        mock_provider.call.return_value = "This is not valid JSON"

        text = """
        Furthermore, the research demonstrates significant findings.
        This study examines multiple factors affecting the outcome.
        """
        extractor = DocumentContextExtractor(llm_provider=mock_provider)
        ctx = extractor.extract(text)

        # Should fall back to heuristics
        assert ctx.tone != ""


class TestExtractDocumentContext:
    """Tests for the convenience function."""

    def test_convenience_function(self):
        """Test the extract_document_context function."""
        text = "This is a simple test document. It has two sentences."
        ctx = extract_document_context(text)

        assert isinstance(ctx, DocumentContext)
        assert ctx.total_paragraphs >= 1
        assert ctx.total_words > 0
