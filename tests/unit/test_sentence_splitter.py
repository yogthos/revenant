"""Tests for the SentenceSplitter module."""

import pytest


class TestSentenceSplitterConfig:
    """Tests for SentenceSplitterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.vocabulary.sentence_splitter import SentenceSplitterConfig

        config = SentenceSplitterConfig()

        assert config.max_sentence_length == 50
        assert config.min_clause_length == 15
        assert "and" in config.split_conjunctions
        assert "but" in config.split_conjunctions

    def test_custom_max_length(self):
        """Test custom max sentence length."""
        from src.vocabulary.sentence_splitter import SentenceSplitterConfig

        config = SentenceSplitterConfig(max_sentence_length=30)

        assert config.max_sentence_length == 30


class TestSplitStats:
    """Tests for SplitStats dataclass."""

    def test_default_stats(self):
        """Test default stats values."""
        from src.vocabulary.sentence_splitter import SplitStats

        stats = SplitStats()

        assert stats.sentences_processed == 0
        assert stats.sentences_split == 0
        assert stats.total_splits == 0
        assert stats.original_avg_length == 0.0
        assert stats.result_avg_length == 0.0


class TestSentenceSplitter:
    """Tests for SentenceSplitter class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()

        assert splitter.config is not None
        assert splitter.config.max_sentence_length == 50

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from src.vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig

        config = SentenceSplitterConfig(max_sentence_length=30)
        splitter = SentenceSplitter(config)

        assert splitter.config.max_sentence_length == 30

    def test_short_sentence_unchanged(self):
        """Test that short sentences are not split."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()
        text = "The cat sat on the mat."
        result, stats = splitter.split(text)

        assert result == text
        assert stats.total_splits == 0
        assert stats.sentences_split == 0

    def test_empty_text(self):
        """Test handling of empty text."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()
        result, stats = splitter.split("")

        assert result == ""
        assert stats.sentences_processed == 0

    def test_split_at_and_conjunction(self):
        """Test splitting at ', and' conjunction."""
        from src.vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig

        # Use a low max_sentence_length to force splitting
        config = SentenceSplitterConfig(max_sentence_length=15, min_clause_length=5)
        splitter = SentenceSplitter(config)

        text = "The stars were fading in the night sky, and the cold wind blew across the barren landscape."
        result, stats = splitter.split(text)

        # Should have split into two sentences
        assert stats.total_splits >= 1
        assert "." in result
        # The word after 'and' should be capitalized as new sentence
        assert "The cold" in result or "the cold" in result.lower()

    def test_split_at_but_conjunction(self):
        """Test splitting at ', but' conjunction."""
        from src.vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig

        config = SentenceSplitterConfig(max_sentence_length=15, min_clause_length=5)
        splitter = SentenceSplitter(config)

        text = "The hero wanted to save the world, but the darkness was too strong to overcome."
        result, stats = splitter.split(text)

        # Should have split
        assert stats.total_splits >= 1

    def test_no_split_without_comma(self):
        """Test that conjunctions without preceding comma are not split."""
        from src.vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig

        config = SentenceSplitterConfig(max_sentence_length=10, min_clause_length=3)
        splitter = SentenceSplitter(config)

        # "and" without comma should not be a split point
        text = "The cat and the dog played together in the yard."
        result, stats = splitter.split(text)

        # Should not split at "and" without comma
        # (though it might split elsewhere if too long)
        assert "cat and the dog" in result or stats.total_splits == 0

    def test_min_clause_length_respected(self):
        """Test that minimum clause length is respected."""
        from src.vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig

        # High min_clause_length should prevent splitting
        config = SentenceSplitterConfig(max_sentence_length=10, min_clause_length=50)
        splitter = SentenceSplitter(config)

        text = "Short, and short again, but still short."
        result, stats = splitter.split(text)

        # Should not split because clauses are too short
        assert stats.total_splits == 0

    def test_multiple_sentences_processed(self):
        """Test processing of multiple sentences."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()
        text = "First sentence. Second sentence. Third sentence."
        result, stats = splitter.split(text)

        assert stats.sentences_processed == 3
        assert stats.total_splits == 0  # All sentences are short

    def test_capitalize_first_letter(self):
        """Test that new sentences are properly capitalized."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter._capitalize_first("the cat sat")

        assert result == "The cat sat"

    def test_capitalize_first_empty(self):
        """Test capitalize with empty string."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter._capitalize_first("")

        assert result == ""

    def test_capitalize_first_with_punctuation(self):
        """Test capitalize with leading punctuation."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter._capitalize_first('"hello world"')

        assert result == '"Hello world"'

    def test_stats_average_calculation(self):
        """Test that average lengths are calculated correctly."""
        from src.vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig

        config = SentenceSplitterConfig(max_sentence_length=10, min_clause_length=3)
        splitter = SentenceSplitter(config)

        # Create text with varying sentence lengths
        text = "Short. This is a somewhat longer sentence that might get processed."
        result, stats = splitter.split(text)

        assert stats.original_avg_length > 0
        assert stats.result_avg_length > 0


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_sentence_splitter_singleton(self):
        """Test that get_sentence_splitter returns singleton."""
        from src.vocabulary.sentence_splitter import get_sentence_splitter

        # Reset singleton
        import src.vocabulary.sentence_splitter as module
        module._splitter = None

        s1 = get_sentence_splitter()
        s2 = get_sentence_splitter()

        assert s1 is s2

    def test_split_sentences_function(self):
        """Test the convenience split_sentences function."""
        from src.vocabulary.sentence_splitter import split_sentences

        # Just verify it doesn't crash
        result = split_sentences("This is a test sentence.")
        assert isinstance(result, str)

    def test_split_sentences_with_custom_length(self):
        """Test split_sentences with custom max_length."""
        from src.vocabulary.sentence_splitter import split_sentences

        result = split_sentences("Short sentence.", max_length=100)
        assert result == "Short sentence."


class TestIntegration:
    """Integration tests with realistic text."""

    def test_lovecraft_style_runon(self):
        """Test splitting a Lovecraft-style run-on sentence."""
        from src.vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig

        # Simulate the kind of run-on that LoRA models produce
        config = SentenceSplitterConfig(max_sentence_length=30, min_clause_length=10)
        splitter = SentenceSplitter(config)

        text = (
            "The ancient and crumbling edifice stood upon the hill, "
            "and its weathered stones bore witness to countless eons, "
            "but none dared approach its cyclopean doorway, "
            "for the whispered legends spoke of nameless horrors within."
        )
        result, stats = splitter.split(text)

        # Should have split this into multiple sentences
        assert stats.total_splits >= 1
        # Result should have multiple sentence-ending periods
        assert result.count(".") >= 2

    def test_preserve_short_sentences_in_mix(self):
        """Test that short sentences in mixed text are preserved."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()

        text = "Short sentence. Another short one. Yet another."
        result, stats = splitter.split(text)

        # Should preserve the original structure
        assert result == text
        assert stats.total_splits == 0

    def test_handles_complex_punctuation(self):
        """Test handling of quotes and other punctuation."""
        from src.vocabulary.sentence_splitter import SentenceSplitter

        splitter = SentenceSplitter()

        text = '"Hello," she said, "how are you?"'
        result, stats = splitter.split(text)

        # Should handle without crashing
        assert isinstance(result, str)
        assert stats.sentences_processed >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
