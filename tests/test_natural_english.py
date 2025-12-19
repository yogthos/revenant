"""Test for natural English phrasing - detect stilted translation-ese.

This test ensures that the fluency checker catches awkward constructions
like interrupted future tense and unnecessary passive voice.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with error handling for missing dependencies
try:
    from src.validator.semantic_critic import SemanticCritic
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")
    # Create dummy class to prevent NameError
    class SemanticCritic:
        def _check_fluency(self, text: str):
            return 0.5, "Dummy feedback"


def test_interrupted_future_tense_detection():
    """Test that interrupted future tense patterns are detected and penalized.

    These stilted patterns should result in low fluency scores:
    - "will, in time, be"
    - "shall, ultimately, succumb"
    - "will, eventually, break"
    """
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_interrupted_future_tense_detection (missing dependencies)")
        return
    critic = SemanticCritic(config_path="config.json")

    # Test cases that should FAIL (stilted patterns)
    stilted_examples = [
        "It will, in due course, be finished.",
        "They shall, strictly speaking, remain.",
        "The star will, eventually, die.",
        "Every object will, in time, be broken.",
        "The system will, in the end, collapse.",
        "We shall, without doubt, succeed.",
    ]

    print("\nTest: Interrupted future tense detection")
    for text in stilted_examples:
        score, feedback = critic._check_fluency(text)
        assert score < 0.8, f"Stilted pattern '{text}' should have low fluency score, got {score}"
        assert "stilted" in feedback.lower() or "interrupt" in feedback.lower() or score < 0.6, \
            f"Stilted pattern '{text}' should be flagged, got feedback: '{feedback}'"
    print("✓ PASSED: Interrupted future tense patterns detected")


def test_natural_english_passes():
    """Test that natural English constructions pass fluency checks.

    These natural patterns should result in high fluency scores:
    - "eventually breaks"
    - "eventually dies"
    - Simple present tense
    """
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_natural_english_passes (missing dependencies)")
        return
    critic = SemanticCritic(config_path="config.json")

    # Test cases that should PASS (natural patterns)
    natural_examples = [
        "It eventually finishes.",
        "The star eventually dies.",
        "The star dies.",
        "Every object eventually breaks.",
        "Objects break.",
        "The star succumbs to erosion.",
        "Every object we touch eventually breaks.",
    ]

    print("\nTest: Natural English passes")
    for text in natural_examples:
        score, feedback = critic._check_fluency(text)
        assert score >= 0.7, f"Natural pattern '{text}' should have high fluency score, got {score} with feedback: '{feedback}'"
    print("✓ PASSED: Natural English patterns pass")


def test_future_passive_voice_penalty():
    """Test that future passive voice is penalized.

    Patterns like "will be broken" should be penalized in favor of active voice.
    """
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_future_passive_voice_penalty (missing dependencies)")
        return
    critic = SemanticCritic(config_path="config.json")

    # Test cases with future passive (should be penalized)
    passive_examples = [
        "Every object will be broken.",
        "The star will be eroded.",
        "The system will be taken.",
        "It will be given.",
    ]

    print("\nTest: Future passive voice penalty")
    for text in passive_examples:
        score, feedback = critic._check_fluency(text)
        # Future passive should reduce score, but not as much as interrupted tense
        assert score < 1.0, f"Future passive '{text}' should be penalized, got {score}"
        # Check that feedback mentions active voice preference
        assert "active" in feedback.lower() or "passive" in feedback.lower() or score < 0.9, \
            f"Future passive '{text}' should mention active voice preference, got feedback: '{feedback}'"
    print("✓ PASSED: Future passive voice penalized")


def test_combined_stilted_patterns():
    """Test that sentences with multiple stilted patterns are heavily penalized."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_combined_stilted_patterns (missing dependencies)")
        return
    critic = SemanticCritic(config_path="config.json")

    # Worst case: interrupted future tense + passive voice
    worst_examples = [
        "Every object will, in time, be broken.",
        "The star will, eventually, be eroded.",
    ]

    print("\nTest: Combined stilted patterns")
    for text in worst_examples:
        score, feedback = critic._check_fluency(text)
        # Combined patterns should result in lower scores than individual patterns
        # Stilted pattern alone: ~0.5, future passive alone: ~0.85, combined should be lower
        assert score < 0.6, f"Combined stilted pattern '{text}' should have low score, got {score}"
        assert len(feedback) > 0, f"Combined stilted pattern '{text}' should have feedback"
    print("✓ PASSED: Combined stilted patterns heavily penalized")


def test_simple_present_for_universal_truths():
    """Test that simple present tense (for universal truths) passes."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_simple_present_for_universal_truths (missing dependencies)")
        return
    critic = SemanticCritic(config_path="config.json")

    # Universal truths in simple present should pass
    universal_truths = [
        "Stars die.",
        "Objects break.",
        "The cycle continues.",
        "Nature follows its course.",
        "Every star eventually succumbs to erosion.",
    ]

    print("\nTest: Simple present for universal truths")
    for text in universal_truths:
        score, feedback = critic._check_fluency(text)
        assert score >= 0.7, f"Universal truth in simple present '{text}' should pass, got {score} with feedback: '{feedback}'"
    print("✓ PASSED: Simple present tense passes")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Natural English Tests")
    print("=" * 60)

    if not DEPENDENCIES_AVAILABLE:
        print(f"\n⚠ All tests skipped due to missing dependencies: {IMPORT_ERROR}")
        print("=" * 60)
        sys.exit(1)

    test_interrupted_future_tense_detection()
    test_natural_english_passes()
    test_future_passive_voice_penalty()
    test_combined_stilted_patterns()
    test_simple_present_for_universal_truths()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

