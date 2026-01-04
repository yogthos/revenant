"""Tests for the prompts module."""

import pytest
from pathlib import Path
import tempfile
import os

from src.utils.prompts import (
    load_prompt,
    format_prompt,
    get_prompt_with_fallback,
    list_prompts,
    clear_prompt_cache,
    PROMPTS_DIR,
)


class TestLoadPrompt:
    """Tests for load_prompt function."""

    def test_load_existing_prompt(self):
        """Test loading an existing prompt file."""
        prompt = load_prompt("style_transfer")
        assert "{author}" in prompt
        assert "{content}" in prompt

    def test_load_nonexistent_prompt_raises_error(self):
        """Test that loading a nonexistent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz")

    def test_prompt_caching(self):
        """Test that prompts are cached."""
        clear_prompt_cache()
        # Load same prompt twice
        prompt1 = load_prompt("style_transfer")
        prompt2 = load_prompt("style_transfer")
        # Should be the same object due to caching
        assert prompt1 is prompt2


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_format_with_variables(self):
        """Test formatting a prompt with variables."""
        prompt = format_prompt(
            "style_transfer",
            author="Carl Sagan",
            content="The universe is vast.",
            structural_guidance="",
        )
        assert "Carl Sagan" in prompt
        assert "The universe is vast." in prompt
        assert "{author}" not in prompt
        assert "{content}" not in prompt

    def test_format_rtt_prompt(self):
        """Test formatting the RTT Mandarin prompt (system prompt)."""
        prompt = format_prompt("rtt_to_mandarin")
        # This is a system prompt with no variables to format
        assert "HSK" in prompt or "translator" in prompt.lower()

    def test_format_with_structural_guidance(self):
        """Test formatting with structural guidance."""
        prompt = format_prompt(
            "style_transfer",
            author="H.P. Lovecraft",
            content="A strange creature appeared.",
            structural_guidance="\n\nRHYTHM: Vary between 5 and 35 words\n",
        )
        assert "H.P. Lovecraft" in prompt
        assert "A strange creature appeared." in prompt
        assert "RHYTHM" in prompt


class TestGetPromptWithFallback:
    """Tests for get_prompt_with_fallback function."""

    def test_returns_file_content_when_exists(self):
        """Test that it returns file content when file exists."""
        fallback = "This is a fallback"
        prompt = get_prompt_with_fallback(
            "repair_system",
            fallback,
        )
        assert fallback not in prompt
        assert len(prompt) > 0

    def test_returns_fallback_when_not_exists(self):
        """Test that it returns fallback when file doesn't exist."""
        fallback = "This is a fallback for {author}"
        prompt = get_prompt_with_fallback(
            "nonexistent_prompt_xyz",
            fallback,
            author="Test Author"
        )
        assert "This is a fallback for Test Author" in prompt


class TestListPrompts:
    """Tests for list_prompts function."""

    def test_list_default_prompts(self):
        """Test listing prompts from default directory."""
        prompts = list_prompts()
        assert len(prompts) > 0
        assert "style_transfer" in prompts
        assert "repair_system" in prompts

    def test_list_prompts_returns_paths(self):
        """Test that list_prompts returns Path objects."""
        prompts = list_prompts()
        for name, path in prompts.items():
            assert isinstance(path, Path)
            assert path.exists()

    def test_list_prompts_empty_directory(self, tmp_path):
        """Test listing prompts from empty directory."""
        prompts = list_prompts(tmp_path)
        assert prompts == {}


class TestClearPromptCache:
    """Tests for clear_prompt_cache function."""

    def test_clear_cache(self):
        """Test that cache can be cleared."""
        # Load a prompt to populate cache
        load_prompt("style_transfer")
        # Clear should not raise
        clear_prompt_cache()
        # Should still be able to load after clear
        prompt = load_prompt("style_transfer")
        assert prompt is not None


class TestPromptFiles:
    """Tests for the actual prompt files."""

    def test_all_prompts_have_valid_placeholders(self):
        """Test that all prompts have valid Python format placeholders."""
        prompts = list_prompts()
        for name, path in prompts.items():
            content = path.read_text()
            # Check that placeholders are valid (no unmatched braces)
            # This is a basic check - actual formatting will validate fully
            open_count = content.count('{')
            close_count = content.count('}')
            assert open_count == close_count, f"Unmatched braces in {name}"

    def test_core_prompts_exist(self):
        """Test that core prompts used by the pipeline exist."""
        required_prompts = [
            "style_transfer",
            "rtt_to_mandarin",
            "rtt_to_english",
            "repair_system",
        ]
        prompts = list_prompts()
        for name in required_prompts:
            assert name in prompts, f"Missing required prompt: {name}"

    def test_prompts_not_empty(self):
        """Test that no prompt files are empty."""
        prompts = list_prompts()
        for name, path in prompts.items():
            content = path.read_text()
            assert len(content.strip()) > 0, f"Empty prompt file: {name}"
