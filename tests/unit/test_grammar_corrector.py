"""Tests for the GrammarCorrector module."""

import pytest
from unittest.mock import MagicMock, patch


class TestGrammarCorrectorConfig:
    """Tests for GrammarCorrectorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.vocabulary.grammar_corrector import GrammarCorrectorConfig

        config = GrammarCorrectorConfig()

        assert config.language == "en-US"
        assert "STYLE" in config.skip_categories
        assert "CASING" in config.skip_categories
        assert "PASSIVE_VOICE" in config.skip_rules
        assert "TOO_LONG_SENTENCE" in config.skip_rules
        assert "GRAMMAR" in config.fix_only_categories
        assert "TYPOS" in config.fix_only_categories

    def test_custom_language(self):
        """Test custom language setting."""
        from src.vocabulary.grammar_corrector import GrammarCorrectorConfig

        config = GrammarCorrectorConfig(language="en-GB")

        assert config.language == "en-GB"


class TestGrammarStats:
    """Tests for GrammarStats dataclass."""

    def test_default_stats(self):
        """Test default stats values."""
        from src.vocabulary.grammar_corrector import GrammarStats

        stats = GrammarStats()

        assert stats.total_matches == 0
        assert stats.filtered_matches == 0
        assert stats.corrections_applied == 0
        assert len(stats.categories_found) == 0
        assert len(stats.rules_found) == 0


class TestGrammarCorrector:
    """Tests for GrammarCorrector class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        assert corrector.config is not None
        assert corrector.config.language == "en-US"

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from src.vocabulary.grammar_corrector import GrammarCorrector, GrammarCorrectorConfig

        config = GrammarCorrectorConfig(language="en-GB")
        corrector = GrammarCorrector(config)

        assert corrector.config.language == "en-GB"

    def test_correct_short_text_skipped(self):
        """Test that very short text is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        # Mock to avoid loading LanguageTool
        corrector._available = True
        corrector._tool = MagicMock()

        text = "Hi."  # Less than min_words (3)
        result, stats = corrector.correct(text)

        assert result == text
        assert stats.corrections_applied == 0
        # Should not have called the tool
        corrector._tool.check.assert_not_called()

    def test_correct_when_unavailable(self):
        """Test that correction is skipped when LanguageTool is unavailable."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = False

        text = "This is a test sentence with grammar errors."
        result, stats = corrector.correct(text)

        assert result == text
        assert stats.corrections_applied == 0

    def test_should_skip_style_category(self):
        """Test that STYLE category is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "STYLE"
        mock_match.rule_id = "SOME_RULE"

        assert corrector._should_skip_match(mock_match) is True

    def test_should_skip_casing_category(self):
        """Test that CASING category is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "CASING"
        mock_match.rule_id = "SOME_RULE"

        assert corrector._should_skip_match(mock_match) is True

    def test_should_skip_passive_voice_rule(self):
        """Test that PASSIVE_VOICE rule is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "GRAMMAR"  # Would normally be fixed
        mock_match.rule_id = "PASSIVE_VOICE"

        assert corrector._should_skip_match(mock_match) is True

    def test_should_not_skip_grammar_category(self):
        """Test that GRAMMAR category is NOT skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "GRAMMAR"
        mock_match.rule_id = "SOME_GRAMMAR_RULE"

        assert corrector._should_skip_match(mock_match) is False

    def test_should_not_skip_typos_category(self):
        """Test that TYPOS category is NOT skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "TYPOS"
        mock_match.rule_id = "SOME_TYPO_RULE"

        assert corrector._should_skip_match(mock_match) is False

    def test_apply_corrections_empty_matches(self):
        """Test applying corrections with no matches."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "This is a test."

        result = corrector._apply_corrections(text, [])

        assert result == text

    def test_apply_corrections_single_match(self):
        """Test applying a single correction."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "The horror were lurking."

        mock_match = MagicMock()
        mock_match.offset = 11  # Position of "were"
        mock_match.error_length = 4  # Length of "were"
        mock_match.replacements = ["was"]

        result = corrector._apply_corrections(text, [mock_match])

        assert result == "The horror was lurking."

    def test_apply_corrections_multiple_matches(self):
        """Test applying multiple corrections in correct order."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "She dont like hte food."

        # "dont" -> "doesn't" at position 4
        match1 = MagicMock()
        match1.offset = 4
        match1.error_length = 4
        match1.replacements = ["doesn't"]

        # "hte" -> "the" at position 14
        match2 = MagicMock()
        match2.offset = 14
        match2.error_length = 3
        match2.replacements = ["the"]

        result = corrector._apply_corrections(text, [match1, match2])

        assert result == "She doesn't like the food."

    def test_analyze_empty_when_unavailable(self):
        """Test that analyze returns empty list when unavailable."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = False

        result = corrector.analyze("This is a test.")

        assert result == []

    def test_correct_handles_exception(self):
        """Test that exceptions during correction are handled gracefully."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = True
        corrector._tool = MagicMock()
        corrector._tool.check.side_effect = Exception("Test error")

        text = "This is a test sentence."
        result, stats = corrector.correct(text)

        # Should return original text on error
        assert result == text
        assert stats.corrections_applied == 0


class TestGrammarCorrectorIntegration:
    """Integration tests that may require LanguageTool."""

    @pytest.fixture
    def corrector(self):
        """Create a grammar corrector for testing."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        # Check if LanguageTool is available
        if not corrector.available:
            pytest.skip("LanguageTool not installed")
        return corrector

    def test_correct_spelling_error(self, corrector):
        """Test correction of spelling errors."""
        # This test only runs if LanguageTool is installed
        text = "The cat sat onn the mat."
        result, stats = corrector.correct(text)

        # Should fix "onn" -> "on"
        assert "on the mat" in result or stats.corrections_applied > 0

    def test_preserve_archaic_language(self, corrector):
        """Test that archaic language is preserved."""
        text = "Whereupon the nameless dread descended upon the village."
        result, stats = corrector.correct(text)

        # Should preserve "Whereupon" - not change it
        assert "Whereupon" in result or "whereupon" in result.lower()

    def test_preserve_long_sentences(self, corrector):
        """Test that long sentences are not flagged."""
        text = (
            "The ancient and crumbling edifice stood silently upon the hill, "
            "its weathered stones bearing witness to countless eons of solitude, "
            "while the eldritch mists swirled about its foundation in patterns "
            "that seemed to defy the natural laws of our comprehension."
        )
        original_words = len(text.split())
        result, stats = corrector.correct(text)

        # Should not drastically change length (TOO_LONG_SENTENCE should be skipped)
        result_words = len(result.split())
        assert abs(result_words - original_words) < 5


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_grammar_corrector_singleton(self):
        """Test that get_grammar_corrector returns singleton."""
        from src.vocabulary.grammar_corrector import get_grammar_corrector

        # Reset singleton
        import src.vocabulary.grammar_corrector as module
        module._corrector = None

        c1 = get_grammar_corrector()
        c2 = get_grammar_corrector()

        assert c1 is c2

    def test_correct_grammar_function(self):
        """Test the convenience correct_grammar function."""
        from src.vocabulary.grammar_corrector import correct_grammar

        # Reset singleton
        import src.vocabulary.grammar_corrector as module
        module._corrector = None

        # Just verify it doesn't crash (LanguageTool may not be installed)
        result = correct_grammar("This is a test.")
        assert isinstance(result, str)


class TestDefaultSkipLists:
    """Tests for the default skip categories and rules."""

    def test_default_skip_categories_complete(self):
        """Test that default skip categories include expected values."""
        from src.vocabulary.grammar_corrector import DEFAULT_SKIP_CATEGORIES

        expected = {"STYLE", "CASING", "MISC", "TYPOGRAPHY", "REDUNDANCY"}
        assert expected.issubset(DEFAULT_SKIP_CATEGORIES)

    def test_default_skip_rules_complete(self):
        """Test that default skip rules include expected values."""
        from src.vocabulary.grammar_corrector import DEFAULT_SKIP_RULES

        expected = {
            "PASSIVE_VOICE",
            "TOO_LONG_SENTENCE",
            "SENTENCE_FRAGMENT",
            "EN_QUOTES",
        }
        assert expected.issubset(DEFAULT_SKIP_RULES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
