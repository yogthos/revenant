"""Contract tests for translate_paragraph method.

These tests define the contract/behavior of translate_paragraph and ensure
it doesn't break when changes are made. If any test fails, the contract
has been violated and functionality is broken.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import re

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.translator import StyleTranslator
from src.analysis.semantic_analyzer import PropositionExtractor
import numpy as np


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, mock_responses=None):
        self.call_count = 0
        self.call_history = []
        self.mock_responses = mock_responses or {}

    def call(self, system_prompt, user_prompt, model_type="editor", require_json=False, temperature=0.7, max_tokens=500, timeout=None, top_p=None):
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_type": model_type,
            "require_json": require_json
        })

        # Default mock response
        if "extract every distinct fact" in user_prompt.lower() or "semantic analyzer" in system_prompt.lower():
            return json.dumps([
                "Human experience reinforces the rule of finitude",
                "The biological cycle defines our reality",
                "Stars eventually die"
            ])
        elif "write a single cohesive paragraph" in user_prompt.lower() or "ghostwriter" in system_prompt.lower():
            return json.dumps([
                "It is through the dialectical process of contradiction and resolution that we come to understand the fundamental relationships between phenomena."
            ])

        return json.dumps(["Mock response"])


class MockStyleAtlas:
    """Mock StyleAtlas for testing."""

    def __init__(self, examples=None):
        self.examples = examples or [
            "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction.",
            "The historical trajectory of human society demonstrates a continuous struggle between opposing forces."
        ]
        self.author_style_vector = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)

    def get_examples_by_rhetoric(self, rhetorical_type, top_k, author_name, query_text=None, exclude=None):
        return self.examples[:top_k]

    def get_author_style_vector(self, author_name: str):
        return self.author_style_vector


def test_translate_paragraph_empty_input():
    """Contract: Empty paragraph input returns empty string.

    This test ensures translate_paragraph handles empty input correctly.
    If this test fails, empty input handling is broken.
    """
    translator = StyleTranslator()
    mock_atlas = MockStyleAtlas()

    result = translator.translate_paragraph(
        paragraph="",
        atlas=mock_atlas,
        author_name="Test Author",
        verbose=False
    )

    assert result == "", "Empty input should return empty string"
    print("✓ Contract: Empty input returns empty")


def test_translate_paragraph_whitespace_only():
    """Contract: Whitespace-only paragraph returns as-is.

    This test ensures translate_paragraph handles whitespace correctly.
    """
    translator = StyleTranslator()
    mock_atlas = MockStyleAtlas()

    result = translator.translate_paragraph(
        paragraph="   \n\t  ",
        atlas=mock_atlas,
        author_name="Test Author",
        verbose=False
    )

    assert result == "   \n\t  ", "Whitespace-only input should return as-is"
    print("✓ Contract: Whitespace-only returns as-is")


def test_translate_paragraph_no_propositions_extracted():
    """Contract: If no propositions extracted, returns original paragraph.

    This test ensures the fallback when proposition extraction fails.
    If this test fails, the fallback mechanism is broken.
    """
    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    # Mock proposition extractor to return empty list
    mock_extractor = Mock()
    mock_extractor.extract_atomic_propositions = Mock(return_value=[])
    translator.proposition_extractor = mock_extractor

    mock_atlas = MockStyleAtlas()
    input_paragraph = "Test paragraph."

    result = translator.translate_paragraph(
        paragraph=input_paragraph,
        atlas=mock_atlas,
        author_name="Test Author",
        verbose=False
    )

    assert result == input_paragraph, "Should return original paragraph when no propositions extracted"
    print("✓ Contract: No propositions → returns original")


def test_translate_paragraph_no_examples_retrieved():
    """Contract: If no examples retrieved, still processes (uses fallback).

    This test ensures graceful handling when atlas returns no examples.
    """
    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    # Mock atlas to return no examples
    mock_atlas = MockStyleAtlas(examples=[])

    # Mock critic to return passing result
    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value
        mock_critic_instance.evaluate.return_value = {
            "pass": True,
            "score": 0.85,
            "proposition_recall": 0.9,
            "style_alignment": 0.8,
            "feedback": "Passed",
            "recall_details": {
                "preserved": [],
                "missing": [],
                "scores": {}
            }
        }

        input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."
        result = translator.translate_paragraph(
            paragraph=input_paragraph,
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=False
        )

        # Should still return something (either generated or original)
        assert result is not None, "Should return result even with no examples"
        assert len(result) > 0, "Result should not be empty"
    print("✓ Contract: No examples → graceful handling")


def test_translate_paragraph_style_dna_extraction_failure():
    """Contract: Style DNA extraction failure doesn't break translate_paragraph.

    This test ensures graceful handling when style DNA extraction fails.
    """
    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Mock style extractor to raise exception
    with patch('src.analyzer.style_extractor.StyleExtractor') as MockExtractor:
        mock_extractor_instance = MockExtractor.return_value
        mock_extractor_instance.extract_style_dna = Mock(side_effect=Exception("Extraction failed"))

        # Mock critic to return passing result
        with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value
            mock_critic_instance.evaluate.return_value = {
                "pass": True,
                "score": 0.85,
                "proposition_recall": 0.9,
                "style_alignment": 0.8,
                "feedback": "Passed",
                "recall_details": {
                    "preserved": [],
                    "missing": [],
                    "scores": {}
                }
            }

            input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."
            result = translator.translate_paragraph(
                paragraph=input_paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                style_dna=None,  # Will trigger extraction
                verbose=False
            )

            # Should still return result (either generated or fallback)
            assert result is not None, "Should handle style DNA extraction failure gracefully"
    print("✓ Contract: Style DNA extraction failure → graceful handling")


def test_translate_paragraph_llm_generation_failure():
    """Contract: LLM generation failure returns original paragraph.

    This test ensures the fallback when LLM generation fails.
    """
    translator = StyleTranslator()

    # Mock LLM to raise exception
    mock_llm = MockLLMProvider()
    mock_llm.call = Mock(side_effect=Exception("LLM generation failed"))
    translator.llm_provider = mock_llm

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()
    input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."

    result = translator.translate_paragraph(
        paragraph=input_paragraph,
        atlas=mock_atlas,
        author_name="Test Author",
        verbose=False
    )

    assert result == input_paragraph, "Should return original paragraph when LLM generation fails"
    print("✓ Contract: LLM generation failure → returns original")


def test_translate_paragraph_citation_cleanup_integration():
    """Contract: Phantom citation removal + restore citations work together.

    This test ensures both citation cleanup steps work correctly together.
    If this test fails, citation cleanup integration is broken.
    """
    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Input with specific citations
    input_paragraph = "Tom Stonier proposed that information is interconvertible with energy[^155]. A rigid pattern repeats itself[^25]."

    # Mock LLM to generate text with phantom citations
    def mock_llm_call(system_prompt, user_prompt, **kwargs):
        # Return text with both valid and phantom citations
        return json.dumps([
            "Tom Stonier proposed that information is interconvertible with energy[^155]. A rigid pattern repeats itself[^25]. This is important[^1]."
        ])

    translator.llm_provider.call = mock_llm_call

    # Mock critic to return passing result
    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value
        mock_critic_instance.evaluate.return_value = {
            "pass": True,
            "score": 0.85,
            "proposition_recall": 0.9,
            "style_alignment": 0.8,
            "feedback": "Passed",
            "recall_details": {
                "preserved": [],
                "missing": [],
                "scores": {}
            }
        }

        result = translator.translate_paragraph(
            paragraph=input_paragraph,
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=False
        )

        # Extract citations from result
        citation_pattern = r'\[\^\d+\]'
        result_citations = set(re.findall(citation_pattern, result))
        input_citations = set(re.findall(citation_pattern, input_paragraph))

        # Should have only input citations (phantoms removed)
        phantom_citations = result_citations - input_citations
        assert len(phantom_citations) == 0, f"Phantom citations should be removed, found: {sorted(phantom_citations)}"

        # Should preserve valid citations
        assert "[^155]" in result or "[^25]" in result, "Valid citations should be preserved"
    print("✓ Contract: Citation cleanup integration works (phantom removal + restore)")


def test_translate_paragraph_all_steps_execute_in_order():
    """Contract: All steps execute in correct order.

    This test verifies the execution order: propositions → examples → style DNA → prompt → generation → evaluation → cleanup.
    """
    translator = StyleTranslator()
    translator.llm_provider = MockLLMProvider()
    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    execution_order = []

    # Track proposition extraction
    original_extract = translator.proposition_extractor.extract_atomic_propositions
    def tracked_extract(text):
        execution_order.append("proposition_extraction")
        return original_extract(text)
    translator.proposition_extractor.extract_atomic_propositions = tracked_extract

    # Track example retrieval
    mock_atlas = MockStyleAtlas()
    original_get_examples = mock_atlas.get_examples_by_rhetoric
    def tracked_get_examples(*args, **kwargs):
        execution_order.append("example_retrieval")
        return original_get_examples(*args, **kwargs)
    mock_atlas.get_examples_by_rhetoric = tracked_get_examples

    # Track style DNA extraction
    with patch('src.analyzer.style_extractor.StyleExtractor') as MockExtractor:
        mock_extractor_instance = MockExtractor.return_value
        def tracked_extract_dna(examples):
            execution_order.append("style_dna_extraction")
            return {"lexicon": ["test"], "tone": "Test", "structure": "Test"}
        mock_extractor_instance.extract_style_dna = tracked_extract_dna

        # Track LLM call (generation)
        original_call = translator.llm_provider.call
        def tracked_call(*args, **kwargs):
            execution_order.append("llm_generation")
            return original_call(*args, **kwargs)
        translator.llm_provider.call = tracked_call

        # Track evaluation
        with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
            mock_critic_instance = MockCritic.return_value
            def tracked_evaluate(*args, **kwargs):
                execution_order.append("evaluation")
                return {
                    "pass": True,
                    "score": 0.85,
                    "proposition_recall": 0.9,
                    "style_alignment": 0.8,
                    "feedback": "Passed",
                    "recall_details": {"preserved": [], "missing": [], "scores": {}}
                }
            mock_critic_instance.evaluate = tracked_evaluate

            input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."
            result = translator.translate_paragraph(
                paragraph=input_paragraph,
                atlas=mock_atlas,
                author_name="Test Author",
                verbose=False
            )

            # Verify execution order
            assert "proposition_extraction" in execution_order, "Proposition extraction should execute"
            assert "example_retrieval" in execution_order, "Example retrieval should execute"
            assert execution_order.index("proposition_extraction") < execution_order.index("example_retrieval"), \
                "Proposition extraction should come before example retrieval"
            assert "llm_generation" in execution_order, "LLM generation should execute"
            assert "evaluation" in execution_order, "Evaluation should execute"
            assert execution_order.index("llm_generation") < execution_order.index("evaluation"), \
                "LLM generation should come before evaluation"

            assert result is not None, "Should return result"
    print("✓ Contract: All steps execute in correct order")


if __name__ == "__main__":
    print("Running translate_paragraph Contract Tests...\n")

    tests = [
        test_translate_paragraph_empty_input,
        test_translate_paragraph_whitespace_only,
        test_translate_paragraph_no_propositions_extracted,
        test_translate_paragraph_no_examples_retrieved,
        test_translate_paragraph_style_dna_extraction_failure,
        test_translate_paragraph_llm_generation_failure,
        test_translate_paragraph_citation_cleanup_integration,
        test_translate_paragraph_all_steps_execute_in_order,
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
        print("\n🎉 All translate_paragraph contract tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)

