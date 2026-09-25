"""Tests for persona prompt_builder module.

Tests cover:
- Bug 2: _load_persona_file("") crashes
- Bug 3: adapter_path not threaded to worldview lookup
"""

import pytest
from unittest.mock import patch


class TestLoadPersonaFile:
    """Tests for _load_persona_file edge cases (Bug 2)."""

    def test_empty_string_returns_empty_result(self):
        """Call with empty string should return empty frames, not crash."""
        from src.persona.prompt_builder import _load_persona_file

        # Clear lru_cache to ensure fresh call
        _load_persona_file.cache_clear()
        result = _load_persona_file("")
        assert result == {"narrative_frames": [], "conceptual_frames": []}

    def test_nonexistent_file_raises(self):
        """Bug M10: Non-empty filename that doesn't exist should raise, not silently
        fall back to default. A typo in config should be surfaced loudly."""
        from src.persona.prompt_builder import _load_persona_file

        _load_persona_file.cache_clear()
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="nonexistent_persona_xyz.txt"):
                _load_persona_file("nonexistent_persona_xyz.txt")


class TestGetPersonaFrame:
    """Tests for _get_persona_frame with adapter_path threading (Bug 3)."""

    @patch('src.persona.prompt_builder._get_worldview_filename')
    @patch('src.persona.prompt_builder._load_persona_file')
    def test_adapter_path_passed_to_worldview_lookup(self, mock_load, mock_get_worldview):
        """_get_persona_frame should pass adapter_path to _get_worldview_filename."""
        from src.persona.prompt_builder import _get_persona_frame

        mock_get_worldview.return_value = "test_worldview.txt"
        mock_load.return_value = {
            "narrative_frames": ["Test narrative frame"],
            "conceptual_frames": ["Test conceptual frame"],
        }

        _get_persona_frame(is_narrative=True, adapter_path="lora_adapters/test")
        mock_get_worldview.assert_called_once_with("lora_adapters/test")

    @patch('src.persona.prompt_builder._get_worldview_filename')
    @patch('src.persona.prompt_builder._load_persona_file')
    def test_adapter_path_none_still_works(self, mock_load, mock_get_worldview):
        """_get_persona_frame should work without adapter_path."""
        from src.persona.prompt_builder import _get_persona_frame

        mock_get_worldview.return_value = "default_persona.txt"
        mock_load.return_value = {
            "narrative_frames": ["Default frame"],
            "conceptual_frames": [],
        }

        result = _get_persona_frame(is_narrative=True)
        mock_get_worldview.assert_called_once_with(None)
        assert result == "Default frame"


class TestBuildPersonaPromptAdapterPath:
    """Tests for build_persona_prompt threading adapter_path (Bug 3)."""

    @patch('src.persona.prompt_builder._get_persona_frame')
    @patch('src.persona.prompt_builder._detect_content_type')
    def test_build_persona_prompt_threads_adapter_path(self, mock_detect, mock_get_frame):
        """build_persona_prompt should pass adapter_path to _get_persona_frame."""
        from src.persona.prompt_builder import build_persona_prompt

        mock_detect.return_value = True  # narrative
        mock_get_frame.return_value = "You are recounting events."

        build_persona_prompt(
            content="Test content for the prompt builder.",
            adapter_path="lora_adapters/test",
        )

        mock_get_frame.assert_called_once_with(True, adapter_path="lora_adapters/test", worldview=None)


class TestConstraintTiers:
    """Tests for constraint tier matching training (Bug 3)."""

    def test_no_structural_constraints_tier(self):
        """_build_constraints should not use a separate STRUCTURAL tier."""
        from src.persona.prompt_builder import _build_constraints
        import src.persona.prompt_builder as pb

        # STRUCTURAL_CONSTRAINTS should not exist as a module-level variable
        assert not hasattr(pb, 'STRUCTURAL_CONSTRAINTS'), (
            "STRUCTURAL_CONSTRAINTS should be removed — training only has 3 tiers"
        )

    def test_constraint_tiers_match_training(self):
        """Constraint tiers should be ALWAYS + FREQUENT + ROTATING only."""
        import src.persona.prompt_builder as pb

        # Should only have these 3 tier lists
        assert hasattr(pb, 'ALWAYS_CONSTRAINTS')
        assert hasattr(pb, 'FREQUENT_CONSTRAINTS')
        assert hasattr(pb, 'ROTATING_CONSTRAINTS')
        assert not hasattr(pb, 'STRUCTURAL_CONSTRAINTS')


class TestConstraintWording:
    """Tests for constraint wording matching training (Bug 4)."""

    def test_rotating_constraints_match_training(self):
        """ROTATING_CONSTRAINTS should match training exactly."""
        from src.persona.prompt_builder import ROTATING_CONSTRAINTS

        expected = [
            "Use fragments. Interrupt yourself with dashes (—).",
            "Let ideas collide without transition words.",
            "Do not explain. Imply.",
            "Use at least one rhetorical question.",
            "Interrupt yourself with a parenthetical thought.",
            "Start the paragraph with a conjunction (But, And, Yet, So).",
            "Be biased. Be opinionated. Do not balance your argument.",
            "Vary sentence lengths dramatically. Follow a long sentence with a short one.",
            "Use concrete nouns instead of abstractions. Not 'the concept' but the thing itself.",
            "End on an image or action, not a summary.",
        ]

        assert ROTATING_CONSTRAINTS == expected, (
            f"ROTATING_CONSTRAINTS doesn't match training.\n"
            f"Missing: {set(expected) - set(ROTATING_CONSTRAINTS)}\n"
            f"Extra: {set(ROTATING_CONSTRAINTS) - set(expected)}"
        )


class TestConceptualFrameFallback:
    """Tests for Bug 2: Conceptual frame fallback must match training."""

    def test_conceptual_fallback_matches_training(self):
        """Fallback conceptual frame must be one of the training defaults."""
        training_defaults = [
            "You are reverse-engineering an alien device. Describe the hidden logic as 'invisible machinery'.",
            "You are a coroner analyzing a system crash. Treat the failure as the universe reclaiming order.",
            "Describe this complex system as a mindless 'Leviathan' made of billions of dumb parts.",
            "State these facts with the absolute, pitiless precision of a machine.",
        ]

        from src.persona.prompt_builder import _get_persona_frame

        # Mock file loading to return empty frames → triggers fallback
        with patch('src.persona.prompt_builder._get_worldview_filename', return_value=""):
            with patch('src.persona.prompt_builder._load_persona_file', return_value={
                "narrative_frames": [], "conceptual_frames": []
            }):
                frame = _get_persona_frame(is_narrative=False)

        assert frame in training_defaults, (
            f"Conceptual fallback '{frame}' not in training defaults: {training_defaults}"
        )


class TestWorldviewLookupLogging:
    """Tests for Bug 4: Silent exception swallowing in worldview lookup."""

    def test_exception_logged_not_swallowed(self):
        """When config loading raises, a warning should be logged."""
        from src.persona.prompt_builder import _get_worldview_filename

        with patch('src.config.get_adapter_config', side_effect=RuntimeError("corrupt config")):
            with patch('src.persona.prompt_builder.logger') as mock_logger:
                result = _get_worldview_filename("some/adapter/path")

        assert result == "default_persona.txt"
        mock_logger.warning.assert_called_once()
        assert "corrupt config" in str(mock_logger.warning.call_args)


class TestGraftingSkeletonStringification:
    """Tests for grafting skeleton being formatted via format_for_prompt(), not __repr__."""

    def test_skeleton_in_prompt_uses_format_for_prompt(self):
        """Skeleton should appear as '[Move1] → [Move2]', not as Python repr."""
        from src.persona.prompt_builder import build_persona_prompt
        from src.rag.skeleton_extractor import ArgumentSkeleton
        from src.rag.structural_grafter import GraftingGuidance

        skeleton = ArgumentSkeleton(
            moves=["Concrete Analogy", "Abstract Claim", "Conclusion"],
            raw="[Concrete Analogy] -> [Abstract Claim] -> [Conclusion]",
        )
        guidance = GraftingGuidance(
            sample_text="Sample author text here.",
            skeleton=skeleton,
        )

        prompt = build_persona_prompt(
            content="Test content for style transfer.",
            target_words=50,
            grafting_guidance=guidance,
        )

        # Should contain the formatted skeleton, not the Python repr
        assert "[Concrete Analogy]" in prompt
        assert "ArgumentSkeleton(" not in prompt, (
            f"Prompt contains Python repr instead of formatted skeleton: {prompt}"
        )


class TestUnusedParameters:
    """Bug: build_persona_prompt accepts vocabulary_palette and persona
    parameters but never uses them."""

    def test_no_unused_parameters_in_build_persona_prompt(self):
        """All parameters should be used in the function body."""
        import inspect
        from src.persona.prompt_builder import build_persona_prompt

        sig = inspect.signature(build_persona_prompt)
        source = inspect.getsource(build_persona_prompt)

        # Remove the function signature to check only the body
        # Find where body starts (after the docstring)
        body_start = source.find('"""', source.find('"""') + 3) + 3
        body = source[body_start:]

        for param_name in sig.parameters:
            if param_name == 'self':
                continue
            assert param_name in body, (
                f"Parameter '{param_name}' is accepted but never used in function body"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
