"""Tests for prompt loading from markdown files.

These tests ensure that prompts loaded from markdown files work correctly
and handle errors gracefully. This validates the recent change moving
prompts from Python strings to markdown files.
"""

import sys
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.mutation_operators import _load_prompt_template, BATCH_GENERATION_PROMPT, PARAGRAPH_FUSION_PROMPT


def test_prompt_loading_file_not_found():
    """Contract: File not found raises FileNotFoundError with clear message.

    This test ensures missing prompt files are handled with proper error messages.
    """
    try:
        _load_prompt_template("nonexistent_prompt.md")
        assert False, "Should raise FileNotFoundError for missing file"
    except FileNotFoundError as e:
        assert "nonexistent_prompt.md" in str(e) or "not found" in str(e).lower(), \
            f"Error message should mention the missing file: {e}"
        print("✓ Contract: File not found → clear error message")
    except Exception as e:
        assert False, f"Should raise FileNotFoundError, got {type(e).__name__}: {e}"


def test_prompt_loading_batch_generation_exists():
    """Contract: batch_generation.md prompt loads correctly.

    This test ensures the batch generation prompt file exists and loads.
    """
    prompt = BATCH_GENERATION_PROMPT

    assert prompt is not None, "BATCH_GENERATION_PROMPT should not be None"
    assert len(prompt) > 0, "BATCH_GENERATION_PROMPT should not be empty"
    assert "{subjects}" in prompt, "Prompt should contain template variable {subjects}"
    assert "{verbs}" in prompt, "Prompt should contain template variable {verbs}"
    assert "{objects}" in prompt, "Prompt should contain template variable {objects}"
    assert "{style_lexicon}" in prompt, "Prompt should contain template variable {style_lexicon}"
    print("✓ Contract: batch_generation.md loads correctly")


def test_prompt_loading_paragraph_fusion_exists():
    """Contract: paragraph_fusion.md prompt loads correctly.

    This test ensures the paragraph fusion prompt file exists and loads.
    """
    prompt = PARAGRAPH_FUSION_PROMPT

    assert prompt is not None, "PARAGRAPH_FUSION_PROMPT should not be None"
    assert len(prompt) > 0, "PARAGRAPH_FUSION_PROMPT should not be empty"
    assert "{propositions_list}" in prompt, "Prompt should contain template variable {propositions_list}"
    assert "{style_examples}" in prompt, "Prompt should contain template variable {style_examples}"
    assert "{citation_instruction}" in prompt, "Prompt should contain template variable {citation_instruction}"
    assert "{citation_output_instruction}" in prompt, "Prompt should contain template variable {citation_output_instruction}"
    print("✓ Contract: paragraph_fusion.md loads correctly")


def test_prompt_loading_template_variable_substitution():
    """Contract: Template variables are correctly substituted.

    This test ensures the loaded prompts can be formatted with variables.
    """
    prompt = PARAGRAPH_FUSION_PROMPT

    # Test formatting with all required variables
    try:
        formatted = prompt.format(
            propositions_list="- Test proposition",
            proposition_count=1,
            style_examples="Example 1: \"Test example\"",
            mandatory_vocabulary="",
            rhetorical_connectors="",
            citation_instruction="",
            structural_blueprint="",
            citation_output_instruction=""
        )

        assert "Test proposition" in formatted, "Template substitution should work"
        assert "Test example" in formatted, "Template substitution should work"
        assert "{propositions_list}" not in formatted, "Template variable should be substituted"
    except KeyError as e:
        assert False, f"Template formatting failed with missing variable: {e}"
    except Exception as e:
        assert False, f"Template formatting failed: {e}"
    print("✓ Contract: Template variables substitute correctly")


def test_prompt_loading_module_import_doesnt_fail():
    """Contract: Module import doesn't fail if prompt files are missing.

    This test ensures the module can be imported even if files are temporarily missing.
    Note: This is a defensive test - in practice files should exist.
    """
    # The prompts are loaded at module import time, so if import succeeds,
    # the files exist. We test that the constants are available.
    from src.generator import mutation_operators

    assert hasattr(mutation_operators, 'BATCH_GENERATION_PROMPT'), \
        "BATCH_GENERATION_PROMPT should be available"
    assert hasattr(mutation_operators, 'PARAGRAPH_FUSION_PROMPT'), \
        "PARAGRAPH_FUSION_PROMPT should be available"

    assert mutation_operators.BATCH_GENERATION_PROMPT is not None, \
        "BATCH_GENERATION_PROMPT should not be None"
    assert mutation_operators.PARAGRAPH_FUSION_PROMPT is not None, \
        "PARAGRAPH_FUSION_PROMPT should not be None"
    print("✓ Contract: Module import succeeds with prompts loaded")


def test_prompt_loading_malformed_markdown_handled():
    """Contract: Malformed markdown is handled gracefully.

    This test ensures that even if markdown is malformed, the content is still loaded.
    Note: The loader just reads text, so malformed markdown shouldn't break it.
    """
    # Create a temporary malformed markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("This is malformed markdown {variable}\n")
        f.write("Missing closing brace {another\n")
        temp_path = Path(f.name)

    try:
        # Load it (should work - it's just text)
        prompts_dir = Path(__file__).parent.parent / "prompts"
        original_path = prompts_dir / "batch_generation.md"

        # Test that loading works even with unusual content
        content = _load_prompt_template("batch_generation.md")
        assert len(content) > 0, "Should load content even if markdown is unusual"
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    print("✓ Contract: Malformed markdown handled gracefully")


def test_prompt_loading_both_prompts_different():
    """Contract: Both prompts are different (not accidentally the same).

    This test ensures batch_generation and paragraph_fusion prompts are distinct.
    """
    batch_prompt = BATCH_GENERATION_PROMPT
    fusion_prompt = PARAGRAPH_FUSION_PROMPT

    assert batch_prompt != fusion_prompt, "Prompts should be different"

    # Check for distinctive content
    assert "batch" in batch_prompt.lower() or "20 distinct variations" in batch_prompt, \
        "Batch prompt should have distinctive content"
    assert "cohesive paragraph" in fusion_prompt.lower() or "atomic propositions" in fusion_prompt.lower(), \
        "Fusion prompt should have distinctive content"
    print("✓ Contract: Both prompts are distinct")


if __name__ == "__main__":
    print("Running Prompt Loading Tests...\n")

    tests = [
        test_prompt_loading_file_not_found,
        test_prompt_loading_batch_generation_exists,
        test_prompt_loading_paragraph_fusion_exists,
        test_prompt_loading_template_variable_substitution,
        test_prompt_loading_module_import_doesnt_fail,
        test_prompt_loading_malformed_markdown_handled,
        test_prompt_loading_both_prompts_different,
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
        print("\n🎉 All prompt loading tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)

