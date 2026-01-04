"""Tests for the inference/transfer pipeline.

Tests cover:
- TransferConfig: Configuration dataclass
- StyleTransfer: Main pipeline orchestration
- Integration with LoRA generator
- RAG context integration
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile


# =============================================================================
# Tests for TransferConfig
# =============================================================================

class TestTransferConfig:
    """Tests for TransferConfig dataclass."""

    def test_default_values(self):
        """Test that default values are sensible."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()

        assert config.max_tokens == 512
        assert config.temperature == 0.4
        assert config.top_p == 0.9
        assert config.verify_entailment is True
        assert config.entailment_threshold == 0.7
        assert config.max_repair_attempts == 3
        assert config.reduce_repetition is True
        assert config.lora_scale == 2.0
        assert config.use_rag is False

    def test_custom_values(self):
        """Test that custom values are applied."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(
            temperature=0.8,
            verify_entailment=False,
            lora_scale=1.5,
            use_rag=True,
            rag_examples=5,
        )

        assert config.temperature == 0.8
        assert config.verify_entailment is False
        assert config.lora_scale == 1.5
        assert config.use_rag is True
        assert config.rag_examples == 5

    def test_perspective_options(self):
        """Test perspective configuration."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(perspective="first_person_singular")
        assert config.perspective == "first_person_singular"

        config2 = TransferConfig(perspective="third_person")
        assert config2.perspective == "third_person"

    def test_expansion_ratios(self):
        """Test expansion ratio configuration."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(
            max_expansion_ratio=2.0,
            target_expansion_ratio=1.5,
        )

        assert config.max_expansion_ratio == 2.0
        assert config.target_expansion_ratio == 1.5


# =============================================================================
# Tests for TransferStats
# =============================================================================

class TestTransferStats:
    """Tests for TransferStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        from src.generation.transfer import TransferStats

        stats = TransferStats()

        assert stats.paragraphs_processed == 0
        assert stats.paragraphs_repaired == 0
        assert stats.words_replaced == 0
        assert stats.total_time_seconds == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.generation.transfer import TransferStats

        stats = TransferStats(
            paragraphs_processed=5,
            paragraphs_repaired=1,
            words_replaced=10,
            total_time_seconds=45.5,
            avg_time_per_paragraph=9.1,
            entailment_scores=[0.8, 0.9, 0.85, 0.75, 0.95],
        )

        d = stats.to_dict()

        assert d["paragraphs_processed"] == 5
        assert d["paragraphs_repaired"] == 1
        assert d["words_replaced"] == 10
        assert d["total_time_seconds"] == 45.5
        assert d["avg_time_per_paragraph"] == 9.1
        assert d["avg_entailment_score"] == 0.85  # Average of scores

    def test_to_dict_empty_scores(self):
        """Test to_dict with empty entailment scores."""
        from src.generation.transfer import TransferStats

        stats = TransferStats()
        d = stats.to_dict()

        assert d["avg_entailment_score"] == 0.0


# =============================================================================
# Tests for StyleTransfer
# =============================================================================

class TestStyleTransfer:
    """Tests for StyleTransfer class."""

    @pytest.fixture
    def mock_generator(self):
        """Create a mock LoRA generator."""
        generator = MagicMock()
        generator.generate.return_value = "This is the styled output text."
        return generator

    @pytest.fixture
    def mock_critic(self):
        """Create a mock critic provider."""
        critic = MagicMock()
        critic.provider_name = "mock"
        critic.call.return_value = "Repaired text here."
        return critic

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_init_with_adapter(self, mock_generator_class, mock_critic):
        """Test initialization with adapter path."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)

        transfer = StyleTransfer(
            adapter_path="lora_adapters/test",
            author_name="Test Author",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.author == "Test Author"
        mock_generator_class.assert_called_once()

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_init_without_adapter(self, mock_generator_class, mock_critic):
        """Test initialization without adapter (base model only)."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)

        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test Author",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.author == "Test Author"

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_init_with_rag_enabled(self, mock_generator_class, mock_critic):
        """Test initialization with RAG enabled."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(
            verify_entailment=False,
            use_rag=True,
            rag_examples=3,
        )

        transfer = StyleTransfer(
            adapter_path="lora_adapters/test",
            author_name="Test Author",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.config.use_rag is True
        assert transfer.config.rag_examples == 3

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_ensure_complete_ending_with_period(self, mock_generator_class, mock_critic):
        """Test that text ending with period is unchanged."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        text = "This is a complete sentence."
        result = transfer._ensure_complete_ending(text)

        assert result == text

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_ensure_complete_ending_adds_period(self, mock_generator_class, mock_critic):
        """Test that incomplete text gets period added."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        text = "This sentence is incomplete and trails off"
        result = transfer._ensure_complete_ending(text)

        assert result.endswith(".")

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_clean_repair_output_basic(self, mock_generator_class, mock_critic):
        """Test that clean_repair_output handles empty and simple text."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Empty text should return empty
        assert transfer._clean_repair_output("") == ""
        assert transfer._clean_repair_output("   ") == ""

        # Normal text should pass through
        text = "This is normal text without any LLM prefixes."
        result = transfer._clean_repair_output(text)
        assert result == text

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_clean_repair_output_strips_whitespace(self, mock_generator_class, mock_critic):
        """Test that clean_repair_output strips whitespace."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        text = "  Some text with leading and trailing spaces.  "
        result = transfer._clean_repair_output(text)
        assert result == "Some text with leading and trailing spaces."

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_transfer_paragraph_skips_short(self, mock_generator_class, mock_critic):
        """Test that short paragraphs are skipped."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(
            verify_entailment=False,
            min_paragraph_words=10,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Very short paragraph (below min_paragraph_words)
        para = "Too short."
        result, score = transfer.transfer_paragraph(para)

        # Should pass through unchanged because it's below min_paragraph_words
        assert result == para
        assert score == 1.0

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_get_partial_results(self, mock_generator_class, mock_critic):
        """Test getting partial results after interruption."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Simulate partial transfer
        transfer._transfer_outputs = ["Para 1", "Para 2"]
        transfer._transfer_stats = MagicMock()
        transfer._transfer_stats.paragraphs_processed = 2
        transfer._transfer_stats.total_time_seconds = 30.0
        transfer._transfer_start_time = 0

        output, stats = transfer.get_partial_results()

        assert "Para 1" in output
        assert "Para 2" in output


# =============================================================================
# Tests for Document Transfer
# =============================================================================

class TestDocumentTransfer:
    """Tests for full document transfer."""

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_transfer_document_basic(self, mock_generator_class):
        """Test basic document transfer with mocked paragraph transfer."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            skip_neutralization=True,
            use_document_context=False,
            min_paragraph_words=5,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Simple document
        doc = "First paragraph with enough words to process properly.\n\nSecond paragraph also with sufficient content."

        # Mock transfer_paragraph to return styled output
        with patch.object(transfer, 'transfer_paragraph', return_value=("Styled output paragraph.", 0.9)):
            output, stats = transfer.transfer_document(doc)

        assert stats.paragraphs_processed == 2
        assert len(output) > 0
        assert "Styled output" in output

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_transfer_document_preserves_headings(self, mock_generator_class):
        """Test that headings are passed through unchanged."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Styled content."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            pass_headings_unchanged=True,
            skip_neutralization=True,
            min_paragraph_words=5,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        doc = "# Heading\n\nParagraph content here with enough words to process properly."

        with patch.object(transfer, 'transfer_paragraph', return_value=("Styled output.", 1.0)):
            output, stats = transfer.transfer_document(doc)

        # Heading should be preserved
        assert "# Heading" in output

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_transfer_document_callback(self, mock_generator_class):
        """Test that progress callback is called."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Output."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            skip_neutralization=True,
            min_paragraph_words=3,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        progress_calls = []

        def on_progress(current, total, status):
            progress_calls.append((current, total, status))

        doc = "First paragraph.\n\nSecond paragraph."

        with patch.object(transfer, 'transfer_paragraph', return_value=("Output.", 1.0)):
            output, stats = transfer.transfer_document(doc, on_progress=on_progress)

        assert len(progress_calls) > 0


# =============================================================================
# Tests for RAG Integration
# =============================================================================

class TestRAGIntegration:
    """Tests for RAG integration with transfer pipeline."""

    @patch('src.generation.transfer.LoRAStyleGenerator')
    @patch('src.generation.transfer.create_rag_context')
    def test_rag_context_loaded_when_enabled(self, mock_create_rag, mock_generator_class):
        """Test that RAG context is loaded when enabled."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Output."
        mock_generator_class.return_value = mock_generator

        mock_rag_context = MagicMock()
        mock_rag_context.has_examples.return_value = True
        mock_rag_context.example_count = 3
        mock_create_rag.return_value = mock_rag_context

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            use_rag=True,
            rag_examples=3,
            skip_neutralization=True,
            min_paragraph_words=3,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        doc = "Paragraph with enough words."

        with patch.object(transfer, 'transfer_paragraph', return_value=("Output.", 1.0)):
            output, stats = transfer.transfer_document(doc)

        # RAG context should have been created
        mock_create_rag.assert_called_once()

    def test_style_rag_context_format(self):
        """Test that StyleRAGContext formats examples correctly."""
        from src.rag.session_context import StyleRAGContext

        context = StyleRAGContext(author="Test")
        context.examples = ["First example passage.", "Second example passage."]

        formatted = context.format_for_prompt()

        assert '[Style Example 1]: "First example passage."' in formatted
        assert '[Style Example 2]: "Second example passage."' in formatted

    def test_rag_context_has_examples(self):
        """Test has_examples method."""
        from src.rag.session_context import StyleRAGContext

        context = StyleRAGContext(author="Test")
        assert context.has_examples() is False

        context.examples = ["Example"]
        assert context.has_examples() is True


# =============================================================================
# Tests for Repetition Reduction Integration
# =============================================================================

class TestRepetitionReduction:
    """Tests for repetition reduction in transfer."""

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_repetition_reducer_applied(self, mock_generator_class):
        """Test that repetition reducer is applied to output."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "The amazing amazing text."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            reduce_repetition=True,
            repetition_threshold=2,
            skip_neutralization=True,
            min_paragraph_words=3,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.repetition_reducer is not None

    @patch('src.generation.transfer.LoRAStyleGenerator')
    def test_repetition_reducer_disabled(self, mock_generator_class):
        """Test that repetition reducer can be disabled."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            reduce_repetition=False,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.repetition_reducer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
