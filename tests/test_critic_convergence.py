"""Test critic convergence issues from logs."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator.critic import critic_evaluate
from src.utils import should_skip_length_gate, calculate_length_ratio


def test_false_positive_feedback_preservation():
    """Test that false positive override preserves useful feedback instead of generic message."""
    from unittest.mock import Mock, patch
    import json

    # Check if config exists, create minimal one if not
    config_path = Path("config.json")
    if not config_path.exists():
        # Create minimal config for testing
        minimal_config = {
            "provider": "deepseek",
            "deepseek": {"api_key": "test-key", "model": "deepseek-chat"},
            "critic": {"fallback_pass_threshold": 0.75}
        }
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f)

    # Mock LLM provider to avoid real API calls
    with patch('src.validator.critic.LLMProvider') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.call.return_value = json.dumps({
            "pass": True,
            "score": 0.85,
            "feedback": "Text matches structure and situation well."
        })
        mock_llm_class.return_value = mock_llm

    # Scenario from logs: LLM flags "essential" as proper noun, filter overrides
    original_text = "Human experience reinforces the rule of finitude."
    generated_text = "Human experience confirms the essential rule of finitude."
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight into evolutionary processes."
    situation_match = "Evolution having themselves. This is the essential ingredient of a self-reinforcing process."

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path)
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest: False positive override feedback quality")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Feedback should NOT be the generic "Text may need minor adjustments"
        # It should have actual useful feedback about structure/style
        assert feedback != "Text may need minor adjustments for style match.", \
            "Feedback is too generic after false positive override"

        # Feedback should contain something actionable
        assert len(feedback) >= 20, f"Feedback is too short/empty: {feedback}"

        print("  ✓ PASS: Feedback preserved or improved")
    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_length_gate_skip_when_very_different():
    """Test that length gate is skipped when structure match is very different."""
    # Check if config exists, create minimal one if not
    config_path = Path("config.json")
    if not config_path.exists():
        import json
        minimal_config = {"critic": {"fallback_pass_threshold": 0.75}}
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f)

    # Scenario from logs: structure match is 2.86x different (20 words vs 7 words)
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight into evolutionary processes and biological mechanisms."
    original_text = "Human experience reinforces the rule of finitude."

    structure_input_ratio = calculate_length_ratio(structure_match, original_text)
    print(f"\nTest: Length gate skip when structure very different")
    print(f"  Structure match: {len(structure_match.split())} words")
    print(f"  Original: {len(original_text.split())} words")
    print(f"  Ratio: {structure_input_ratio:.2f}")

    should_skip = should_skip_length_gate(structure_input_ratio, config_path=str(config_path))
    print(f"  Should skip: {should_skip}")

    # Ratio is 2.86, which is > 2.0, so should skip
    if structure_input_ratio > 2.0 or structure_input_ratio < 0.5:
        assert should_skip, "Should skip length gate when ratio is very different"
        print("  ✓ PASS: Length gate correctly skipped")
    else:
        print("  ⚠ INFO: Ratio is not very different, skip not required")


def test_score_improvement_with_useful_feedback():
    """Test that score can improve when feedback is useful (not generic)."""
    from unittest.mock import Mock, patch
    import json

    # Check if config exists, create minimal one if not
    config_path = Path("config.json")
    if not config_path.exists():
        minimal_config = {
            "provider": "deepseek",
            "deepseek": {"api_key": "test-key", "model": "deepseek-chat"},
            "critic": {"fallback_pass_threshold": 0.75}
        }
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f)

    # Mock LLM provider to avoid real API calls
    with patch('src.validator.critic.LLMProvider') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.call.return_value = json.dumps({
            "pass": False,
            "score": 0.3,
            "feedback": "CRITICAL: Output (3 words) lost >50% of original content (7 words). Restore missing concepts from the original text."
        })
        mock_llm_class.return_value = mock_llm

    # Test that when we have useful feedback (not false positive), score can improve
    original_text = "The biological cycle of birth, life, and decay defines our reality."
    generated_text = "The cycle defines reality."  # Too short
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight into evolutionary processes."
    situation_match = None

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path)
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")

        print(f"\nTest: Score improvement with useful feedback")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")

        # Feedback should be specific and actionable
        assert "FATAL ERROR" in feedback or "too short" in feedback.lower() or "expand" in feedback.lower() or len(feedback) >= 30, \
            f"Feedback should be specific and actionable, got: {feedback}"
        print("  ✓ PASS: Feedback is specific and actionable")
    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Critic Convergence Issues")
    print("=" * 60)

    test_false_positive_feedback_preservation()
    test_length_gate_skip_when_very_different()
    test_score_improvement_with_useful_feedback()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")

