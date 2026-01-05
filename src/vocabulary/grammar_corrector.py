"""Style-safe grammar correction using LanguageTool.

Fixes objective grammar and spelling errors while preserving authorial style.
Uses rule-based correction rather than generative models to avoid rewriting.

Key principle: Fix "The horror were lurking" → "The horror was lurking"
but preserve "It was a lugubrious and eldritch night whereupon..."
"""

from dataclasses import dataclass, field
from typing import Set, List, Tuple, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Default categories to skip (preserve author style)
DEFAULT_SKIP_CATEGORIES = {
    "STYLE",           # Don't enforce "plain English" style
    "CASING",          # Don't change author's capitalization choices
    "MISC",            # Miscellaneous style suggestions
    "TYPOGRAPHY",      # Don't enforce typographic conventions
    "REDUNDANCY",      # Authors may use redundancy for effect
}

# Default rules to skip (common style features)
DEFAULT_SKIP_RULES = {
    # Sentence structure
    "PASSIVE_VOICE",           # Many authors use passive deliberately
    "TOO_LONG_SENTENCE",       # Long sentences are a style feature
    "SENTENCE_FRAGMENT",       # Fragments used for effect
    "COMMA_COMPOUND_SENTENCE", # Long compound sentences are valid style

    # Punctuation style
    "EN_QUOTES",               # Don't force smart quotes
    "DASH_RULE",               # Don't enforce dash conventions
    "OXFORD_COMMA",            # Author's choice

    # Word choice
    "VERY",                    # "very" is fine for emphasis
    "RATHER",                  # Stylistic choice
    "SOMEWHAT",                # Stylistic choice

    # Archaic/formal language
    "WHEREAS",                 # Valid formal connector
    "WHILST",                  # Valid British/archaic form
    "AMONGST",                 # Valid British/archaic form
    "FURTHERMORE",             # Valid connector (even if overused by LLMs)
}


@dataclass
class GrammarStats:
    """Statistics from grammar correction."""
    total_matches: int = 0
    filtered_matches: int = 0
    corrections_applied: int = 0
    categories_found: Set[str] = field(default_factory=set)
    rules_found: Set[str] = field(default_factory=set)


@dataclass
class GrammarCorrectorConfig:
    """Configuration for grammar correction."""
    language: str = "en-US"  # or "en-GB" for British English
    skip_categories: Set[str] = field(default_factory=lambda: DEFAULT_SKIP_CATEGORIES.copy())
    skip_rules: Set[str] = field(default_factory=lambda: DEFAULT_SKIP_RULES.copy())

    # Only fix these specific categories (if empty, fix all not skipped)
    fix_only_categories: Set[str] = field(default_factory=lambda: {"GRAMMAR", "TYPOS"})

    # Minimum word count to apply correction (skip very short text)
    min_words: int = 3


class GrammarCorrector:
    """Style-safe grammar corrector using LanguageTool.

    Uses rule-based correction to fix objective errors while preserving
    authorial voice. Does NOT rewrite sentences - only applies targeted fixes.
    """

    def __init__(self, config: Optional[GrammarCorrectorConfig] = None):
        """Initialize the grammar corrector.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or GrammarCorrectorConfig()
        self._tool = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if LanguageTool is available."""
        if self._available is None:
            try:
                import language_tool_python
                self._available = True
            except ImportError:
                self._available = False
                logger.warning(
                    "language-tool-python not installed. "
                    "Grammar correction disabled. "
                    "Install with: pip install language-tool-python"
                )
        return self._available

    @property
    def tool(self):
        """Lazy load LanguageTool."""
        if self._tool is None and self.available:
            import language_tool_python
            logger.info(f"Loading LanguageTool ({self.config.language})...")
            self._tool = language_tool_python.LanguageTool(self.config.language)
            logger.info("LanguageTool loaded successfully")
        return self._tool

    def _should_skip_match(self, match) -> bool:
        """Determine if a match should be skipped.

        Args:
            match: LanguageTool match object.

        Returns:
            True if the match should be skipped.
        """
        # Skip by category
        if match.category in self.config.skip_categories:
            return True

        # Skip by rule ID (use snake_case attribute name)
        if match.rule_id in self.config.skip_rules:
            return True

        # If fix_only_categories is set, only fix those
        if self.config.fix_only_categories:
            if match.category not in self.config.fix_only_categories:
                return True

        return False

    def _apply_corrections(self, text: str, matches: List) -> str:
        """Apply corrections from matches to text.

        Applies corrections in reverse order to preserve offsets.

        Args:
            text: Original text.
            matches: List of filtered matches to apply.

        Returns:
            Corrected text.
        """
        if not matches:
            return text

        # Sort by offset descending to apply from end to start
        sorted_matches = sorted(matches, key=lambda m: m.offset, reverse=True)

        result = text
        for match in sorted_matches:
            # Only apply if there's a replacement suggestion
            if match.replacements:
                replacement = match.replacements[0]  # Use first suggestion
                start = match.offset
                end = match.offset + match.error_length  # snake_case attribute
                result = result[:start] + replacement + result[end:]

        return result

    def correct(self, text: str) -> Tuple[str, GrammarStats]:
        """Apply style-safe grammar correction to text.

        Args:
            text: Text to correct.

        Returns:
            Tuple of (corrected_text, stats).
        """
        stats = GrammarStats()

        # Skip if too short
        if len(text.split()) < self.config.min_words:
            return text, stats

        # Skip if LanguageTool not available
        if not self.available or self.tool is None:
            return text, stats

        try:
            # Get all matches
            matches = self.tool.check(text)
            stats.total_matches = len(matches)

            # Track what we found
            for match in matches:
                stats.categories_found.add(match.category)
                stats.rules_found.add(match.rule_id)

            # Filter matches
            filtered = [m for m in matches if not self._should_skip_match(m)]
            stats.filtered_matches = len(filtered)

            if not filtered:
                logger.debug(f"No grammar corrections needed ({stats.total_matches} style issues skipped)")
                return text, stats

            # Apply corrections
            corrected = self._apply_corrections(text, filtered)
            stats.corrections_applied = len(filtered)

            if corrected != text:
                logger.debug(
                    f"Grammar correction: {stats.corrections_applied} fixes applied "
                    f"({stats.total_matches - stats.filtered_matches} style issues preserved)"
                )

            return corrected, stats

        except Exception as e:
            logger.warning(f"Grammar correction failed: {e}")
            return text, stats

    def analyze(self, text: str) -> List[dict]:
        """Analyze text for grammar issues without correcting.

        Useful for debugging and understanding what LanguageTool detects.

        Args:
            text: Text to analyze.

        Returns:
            List of issue dictionaries with details.
        """
        if not self.available or self.tool is None:
            return []

        try:
            matches = self.tool.check(text)
            return [
                {
                    "rule_id": m.rule_id,
                    "category": m.category,
                    "message": m.message,
                    "context": m.context,
                    "offset": m.offset,
                    "length": m.error_length,
                    "replacements": m.replacements[:3],  # Top 3 suggestions
                    "would_skip": self._should_skip_match(m),
                }
                for m in matches
            ]
        except Exception as e:
            logger.warning(f"Grammar analysis failed: {e}")
            return []

    def close(self):
        """Close the LanguageTool connection."""
        if self._tool is not None:
            try:
                self._tool.close()
            except Exception:
                pass
            self._tool = None


# Module singleton
_corrector: Optional[GrammarCorrector] = None


def get_grammar_corrector(config: Optional[GrammarCorrectorConfig] = None) -> GrammarCorrector:
    """Get or create singleton grammar corrector instance."""
    global _corrector
    if _corrector is None:
        _corrector = GrammarCorrector(config)
    return _corrector


def correct_grammar(text: str) -> str:
    """Convenience function to correct grammar in text.

    Args:
        text: Text to correct.

    Returns:
        Corrected text.
    """
    corrector = get_grammar_corrector()
    corrected, _ = corrector.correct(text)
    return corrected
