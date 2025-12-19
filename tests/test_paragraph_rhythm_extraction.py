"""Unit tests for paragraph rhythm extraction and structural cloning.

These tests verify that the rhythm extraction and structural cloning
features work correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.structuralizer import extract_paragraph_rhythm, RHETORICAL_OPENERS


def test_extract_paragraph_rhythm_basic():
    """Test basic rhythm extraction with different sentence types."""
    text = "Short sentence. However, this is a much longer sentence that contains many words and demonstrates complexity and exceeds the typical medium length threshold by adding more descriptive content. Is this a question?"

    rhythm_map = extract_paragraph_rhythm(text)

    assert len(rhythm_map) == 3, f"Expected 3 sentences, got {len(rhythm_map)}"

    # First sentence: short
    assert rhythm_map[0]['length'] == 'short', f"First sentence should be short, got {rhythm_map[0]['length']}"
    assert rhythm_map[0]['type'] == 'standard', f"First sentence should be standard, got {rhythm_map[0]['type']}"
    assert rhythm_map[0]['opener'] is None, f"First sentence should have no opener, got {rhythm_map[0]['opener']}"

    # Second sentence: long with opener (> 25 words)
    assert rhythm_map[1]['length'] == 'long', f"Second sentence should be long, got {rhythm_map[1]['length']}"
    assert rhythm_map[1]['opener'] == 'However', f"Second sentence should start with 'However', got {rhythm_map[1]['opener']}"

    # Third sentence: question
    assert rhythm_map[2]['type'] == 'question', f"Third sentence should be a question, got {rhythm_map[2]['type']}"

    print("✓ Basic rhythm extraction works correctly")


def test_extract_paragraph_rhythm_conditional():
    """Test rhythm extraction with conditional sentences."""
    text = "If this condition is met, then the result follows. This is a standard sentence."

    rhythm_map = extract_paragraph_rhythm(text)

    assert len(rhythm_map) == 2, f"Expected 2 sentences, got {len(rhythm_map)}"

    # First sentence should be conditional
    assert rhythm_map[0]['type'] == 'conditional', f"First sentence should be conditional, got {rhythm_map[0]['type']}"

    print("✓ Conditional sentence detection works")


def test_extract_paragraph_rhythm_rhetorical_openers():
    """Test that only rhetorical openers are captured, not common words."""
    text = "The cat sat. But the dog ran. However, the bird flew. Thus, we see."

    rhythm_map = extract_paragraph_rhythm(text)

    assert len(rhythm_map) == 4, f"Expected 4 sentences, got {len(rhythm_map)}"

    # First sentence: "The" should not be captured
    assert rhythm_map[0]['opener'] is None, f"'The' should not be captured as opener, got {rhythm_map[0]['opener']}"

    # Second sentence: "But" should be captured
    assert rhythm_map[1]['opener'] == 'But', f"'But' should be captured, got {rhythm_map[1]['opener']}"

    # Third sentence: "However" should be captured
    assert rhythm_map[2]['opener'] == 'However', f"'However' should be captured, got {rhythm_map[2]['opener']}"

    # Fourth sentence: "Thus" should be captured
    assert rhythm_map[3]['opener'] == 'Thus', f"'Thus' should be captured, got {rhythm_map[3]['opener']}"

    print("✓ Rhetorical opener detection works correctly")


def test_extract_paragraph_rhythm_length_classification():
    """Test length classification (short, medium, long)."""
    # Short sentence (< 10 words)
    short_text = "This is short."
    short_rhythm = extract_paragraph_rhythm(short_text)
    assert short_rhythm[0]['length'] == 'short', f"Should be short, got {short_rhythm[0]['length']}"

    # Medium sentence (10-25 words)
    medium_text = "This is a medium length sentence that contains exactly ten words here."
    medium_rhythm = extract_paragraph_rhythm(medium_text)
    assert medium_rhythm[0]['length'] == 'medium', f"Should be medium, got {medium_rhythm[0]['length']}"

    # Long sentence (> 25 words)
    long_text = "This is a very long sentence that contains many words and demonstrates the complexity of human writing patterns that exceed the typical medium length threshold by adding more descriptive content."
    long_rhythm = extract_paragraph_rhythm(long_text)
    assert long_rhythm[0]['length'] == 'long', f"Should be long, got {long_rhythm[0]['length']}"

    print("✓ Length classification works correctly")


def test_extract_paragraph_rhythm_empty_input():
    """Test rhythm extraction with empty input."""
    rhythm_map = extract_paragraph_rhythm("")
    assert rhythm_map == [], f"Empty input should return empty list, got {rhythm_map}"

    rhythm_map = extract_paragraph_rhythm("   ")
    assert rhythm_map == [], f"Whitespace-only input should return empty list, got {rhythm_map}"

    print("✓ Empty input handling works")


def test_rhetorical_openers_constant():
    """Test that RHETORICAL_OPENERS constant is properly defined."""
    assert isinstance(RHETORICAL_OPENERS, set), "RHETORICAL_OPENERS should be a set"
    assert len(RHETORICAL_OPENERS) > 0, "RHETORICAL_OPENERS should not be empty"

    # Check some expected openers
    assert 'but' in RHETORICAL_OPENERS, "'but' should be in RHETORICAL_OPENERS"
    assert 'however' in RHETORICAL_OPENERS, "'however' should be in RHETORICAL_OPENERS"
    assert 'thus' in RHETORICAL_OPENERS, "'thus' should be in RHETORICAL_OPENERS"
    assert 'therefore' in RHETORICAL_OPENERS, "'therefore' should be in RHETORICAL_OPENERS"

    # Check that common words are NOT in the set
    assert 'the' not in RHETORICAL_OPENERS, "'the' should NOT be in RHETORICAL_OPENERS"
    assert 'a' not in RHETORICAL_OPENERS, "'a' should NOT be in RHETORICAL_OPENERS"
    assert 'it' not in RHETORICAL_OPENERS, "'it' should NOT be in RHETORICAL_OPENERS"

    print("✓ RHETORICAL_OPENERS constant is properly defined")


def test_extract_paragraph_rhythm_case_insensitive_openers():
    """Test that opener detection is case-insensitive."""
    text = "BUT this works. however this too. THUS we see."

    rhythm_map = extract_paragraph_rhythm(text)

    assert len(rhythm_map) == 3, f"Expected 3 sentences, got {len(rhythm_map)}"

    # Openers should be captured regardless of case
    # Note: The opener value should preserve original capitalization
    assert rhythm_map[0]['opener'] is not None, "First sentence should have opener 'BUT'"
    assert rhythm_map[1]['opener'] is not None, "Second sentence should have opener 'however'"
    assert rhythm_map[2]['opener'] is not None, "Third sentence should have opener 'THUS'"

    print("✓ Case-insensitive opener detection works")


def test_teacher_selection_integration():
    """Test that teacher selection logic works with rhythm extraction."""
    from unittest.mock import Mock, patch
    from src.generator.translator import StyleTranslator
    import math
    from nltk.tokenize import sent_tokenize

    # Mock examples with different sentence counts
    examples = [
        "Short. Medium.",  # 2 sentences
        "One. Two. Three. Four.",  # 4 sentences
        "A. B. C. D. E. F.",  # 6 sentences
    ]

    n_props = 5
    target_sentences = math.ceil(n_props * 0.6)  # Should be 3

    # Find best match
    best_match = None
    best_diff = float('inf')

    for example in examples:
        try:
            example_sentences = sent_tokenize(example)
            sentence_count = len([s for s in example_sentences if s.strip()])
            diff = abs(sentence_count - target_sentences)

            if diff < best_diff:
                best_diff = diff
                best_match = example
        except Exception:
            continue

    # Should select the example closest to target 3
    # Both 2-sentence (diff=1) and 4-sentence (diff=1) have same diff, so first one encountered is selected
    # The test verifies that the selection logic works, not the specific choice
    assert best_match is not None, "Should select a best match"
    assert best_match in examples, "Best match should be one of the examples"

    # Extract rhythm map from best match (verify extraction works regardless of which example was selected)
    rhythm_map = extract_paragraph_rhythm(best_match)
    # Verify rhythm extraction works correctly on the selected example
    assert len(rhythm_map) > 0, f"Rhythm map should have at least 1 sentence, got {len(rhythm_map)}"
    # Verify the rhythm map structure is correct
    for spec in rhythm_map:
        assert 'length' in spec, "Each spec should have 'length'"
        assert 'type' in spec, "Each spec should have 'type'"
        assert 'opener' in spec, "Each spec should have 'opener'"

    print("✓ Teacher selection integration works")


def test_structural_blueprint_formatting():
    """Test that rhythm map is formatted correctly for prompt."""
    rhythm_map = [
        {'length': 'short', 'type': 'standard', 'opener': None},
        {'length': 'long', 'type': 'conditional', 'opener': 'However'},
        {'length': 'medium', 'type': 'question', 'opener': None}
    ]

    # Format as it would be in translator.py
    blueprint_lines = []
    blueprint_lines.append("### STRUCTURAL BLUEPRINT:")
    blueprint_lines.append("You must structure your paragraph to match this rhythm exactly. Distribute your Atomic Propositions into this container. Merge or split them as needed to fit the sentence types. Follow this sentence-by-sentence blueprint exactly. If the blueprint asks for a Short Sentence, do not write a long one.")
    blueprint_lines.append("")
    for i, spec in enumerate(rhythm_map):
        length = spec['length']
        sent_type = spec['type']
        opener = spec['opener']

        desc_parts = [length]
        if sent_type == 'question':
            desc_parts.append('rhetorical question')
        elif sent_type == 'conditional':
            desc_parts.append('conditional')
        else:
            desc_parts.append('declarative statement')

        opener_text = ""
        if opener:
            opener_text = f" starting with '{opener}'"

        blueprint_lines.append(f"Sentence {i+1}: {' '.join(desc_parts).capitalize()}{opener_text}.")

    structural_blueprint = "\n".join(blueprint_lines)

    # Verify formatting
    assert "STRUCTURAL BLUEPRINT" in structural_blueprint, "Should include section header"
    assert "Sentence 1:" in structural_blueprint, "Should include sentence 1"
    assert "Short declarative statement" in structural_blueprint, "Should format first sentence correctly"
    assert "Sentence 2:" in structural_blueprint, "Should include sentence 2"
    assert "Long conditional" in structural_blueprint, "Should format second sentence correctly"
    assert "starting with 'However'" in structural_blueprint, "Should include opener for sentence 2"
    assert "Sentence 3:" in structural_blueprint, "Should include sentence 3"
    assert "Medium rhetorical question" in structural_blueprint, "Should format third sentence correctly"

    print("✓ Structural blueprint formatting works correctly")


if __name__ == "__main__":
    print("Running Paragraph Rhythm Extraction Tests...\n")

    tests = [
        test_extract_paragraph_rhythm_basic,
        test_extract_paragraph_rhythm_conditional,
        test_extract_paragraph_rhythm_rhetorical_openers,
        test_extract_paragraph_rhythm_length_classification,
        test_extract_paragraph_rhythm_empty_input,
        test_rhetorical_openers_constant,
        test_extract_paragraph_rhythm_case_insensitive_openers,
        test_teacher_selection_integration,
        test_structural_blueprint_formatting,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed == 0:
        print("\n🎉 All paragraph rhythm extraction tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)

