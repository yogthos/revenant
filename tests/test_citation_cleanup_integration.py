"""Integration tests for citation cleanup in translate_paragraph.

These tests ensure that phantom citation removal and citation restoration
work together correctly. This is critical for the recent phantom citation fix.
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
from src.ingestion.blueprint import BlueprintExtractor
import numpy as np


class MockLLMProvider:
    """Mock LLM provider that generates text with phantom citations."""

    def __init__(self):
        self.call_count = 0

    def call(self, system_prompt, user_prompt, model_type="editor", require_json=False, temperature=0.7, max_tokens=500, timeout=None, top_p=None):
        self.call_count += 1

        if "extract every distinct fact" in user_prompt.lower():
            return json.dumps([
                "Tom Stonier proposed that information is interconvertible with energy[^155]",
                "A rigid pattern repeats itself[^25]"
            ])
        elif "write a single cohesive paragraph" in user_prompt.lower():
            # Generate text with BOTH valid and phantom citations
            return json.dumps([
                "Tom Stonier proposed that information is interconvertible with energy[^155]. A rigid pattern repeats itself[^25]. This demonstrates the principle[^1]. The system operates[^2]. Further evidence suggests[^3]."
            ])

        return json.dumps(["Mock response"])


class MockStyleAtlas:
    """Mock StyleAtlas for testing."""

    def __init__(self):
        self.examples = [
            "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction.",
            "The historical trajectory of human society demonstrates a continuous struggle between opposing forces."
        ]
        self.author_style_vector = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)

    def get_examples_by_rhetoric(self, rhetorical_type, top_k, author_name, query_text=None, exclude=None):
        return self.examples[:top_k]

    def get_author_style_vector(self, author_name: str):
        return self.author_style_vector


def test_phantom_removal_and_restore_work_together():
    """Contract: Phantom removal + restore citations work together correctly.

    This test ensures both cleanup steps execute and work together.
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

    # Input with specific citations [^155] and [^25]
    input_paragraph = "Tom Stonier proposed that information is interconvertible with energy[^155]. A rigid pattern repeats itself[^25]."

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

        # Extract citations from input and output
        citation_pattern = r'\[\^\d+\]'
        input_citations = set(re.findall(citation_pattern, input_paragraph))
        output_citations = set(re.findall(citation_pattern, result))

        # Verify phantom citations are removed
        phantom_citations = output_citations - input_citations
        assert len(phantom_citations) == 0, \
            f"Phantom citations should be removed. Found: {sorted(phantom_citations)}, Expected only: {sorted(input_citations)}"

        # Verify valid citations are preserved (at least one should be present)
        assert len(output_citations & input_citations) > 0, \
            f"Valid citations should be preserved. Input had: {sorted(input_citations)}, Output has: {sorted(output_citations)}"
    print("✓ Contract: Phantom removal + restore work together")


def test_citations_at_sentence_boundaries():
    """Contract: Citations at sentence boundaries handled correctly.

    This test ensures citations at the start/end of sentences are handled correctly.
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

    # Input with citation at end
    input_paragraph = "Tom Stonier proposed that information is interconvertible with energy[^155]."

    # Mock critic
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

        # Citation should be preserved
        assert "[^155]" in result, "Citation at sentence end should be preserved"

        # No phantom citations
        citation_pattern = r'\[\^\d+\]'
        input_citations = set(re.findall(citation_pattern, input_paragraph))
        output_citations = set(re.findall(citation_pattern, result))
        phantom_citations = output_citations - input_citations
        assert len(phantom_citations) == 0, f"Should have no phantom citations, found: {sorted(phantom_citations)}"
    print("✓ Contract: Citations at sentence boundaries handled correctly")


def test_multiple_phantom_citations_removed():
    """Contract: Multiple phantom citations are all removed.

    This test ensures the cleanup handles multiple phantoms correctly.
    """
    translator = StyleTranslator()

    # Mock LLM to generate many phantom citations
    def mock_llm_call(system_prompt, user_prompt, **kwargs):
        if "extract every distinct fact" in user_prompt.lower():
            return json.dumps([
                "Tom Stonier proposed that information is interconvertible with energy[^155]"
            ])
        elif "write a single cohesive paragraph" in user_prompt.lower():
            # Generate with many phantom citations
            return json.dumps([
                "Tom Stonier proposed that information is interconvertible with energy[^155]. This is important[^1]. The system works[^2]. Further evidence[^3]. More data[^4]."
            ])
        return json.dumps(["Mock"])

    translator.llm_provider = Mock()
    translator.llm_provider.call = mock_llm_call

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Input with only [^155]
    input_paragraph = "Tom Stonier proposed that information is interconvertible with energy[^155]."

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

        # Extract citations
        citation_pattern = r'\[\^\d+\]'
        input_citations = set(re.findall(citation_pattern, input_paragraph))
        output_citations = set(re.findall(citation_pattern, result))
        phantom_citations = output_citations - input_citations

        # All phantoms should be removed
        assert len(phantom_citations) == 0, \
            f"All phantom citations should be removed. Found: {sorted(phantom_citations)}"

        # Valid citation should be preserved
        assert "[^155]" in result, "Valid citation [^155] should be preserved"
    print("✓ Contract: Multiple phantom citations all removed")


def test_valid_citations_preserved_after_phantom_removal():
    """Contract: Valid citations are preserved even after phantom removal.

    This test ensures valid citations aren't accidentally removed during cleanup.
    """
    translator = StyleTranslator()

    # Mock LLM to generate both valid and phantom citations
    def mock_llm_call(system_prompt, user_prompt, **kwargs):
        if "extract every distinct fact" in user_prompt.lower():
            return json.dumps([
                "Tom Stonier proposed that information is interconvertible with energy[^155]",
                "A rigid pattern repeats itself[^25]"
            ])
        elif "write a single cohesive paragraph" in user_prompt.lower():
            return json.dumps([
                "Tom Stonier proposed that information is interconvertible with energy[^155]. A rigid pattern repeats itself[^25]. This demonstrates[^1]. The system[^2]."
            ])
        return json.dumps(["Mock"])

    translator.llm_provider = Mock()
    translator.llm_provider.call = mock_llm_call

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Input with [^155] and [^25]
    input_paragraph = "Tom Stonier proposed that information is interconvertible with energy[^155]. A rigid pattern repeats itself[^25]."

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

        # Both valid citations should be preserved
        assert "[^155]" in result, "Valid citation [^155] should be preserved"
        assert "[^25]" in result, "Valid citation [^25] should be preserved"

        # No phantom citations
        citation_pattern = r'\[\^\d+\]'
        input_citations = set(re.findall(citation_pattern, input_paragraph))
        output_citations = set(re.findall(citation_pattern, result))
        phantom_citations = output_citations - input_citations
        assert len(phantom_citations) == 0, f"Should have no phantoms, found: {sorted(phantom_citations)}"
    print("✓ Contract: Valid citations preserved after phantom removal")


def test_citation_cleanup_with_no_input_citations():
    """Contract: Citation cleanup works correctly when input has no citations.

    This test ensures cleanup doesn't break when there are no citations to preserve.
    """
    translator = StyleTranslator()

    # Mock LLM to generate phantom citations even when input has none
    def mock_llm_call(system_prompt, user_prompt, **kwargs):
        if "extract every distinct fact" in user_prompt.lower():
            return json.dumps([
                "Human experience reinforces the rule of finitude",
                "The biological cycle defines our reality"
            ])
        elif "write a single cohesive paragraph" in user_prompt.lower():
            # LLM generates phantoms even though input has none
            return json.dumps([
                "Human experience reinforces the rule of finitude[^1]. The biological cycle defines our reality[^2]."
            ])
        return json.dumps(["Mock"])

    translator.llm_provider = Mock()
    translator.llm_provider.call = mock_llm_call

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.7,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Input with NO citations
    input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."

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

        # Should have NO citations in output (all phantoms removed)
        citation_pattern = r'\[\^\d+\]'
        output_citations = set(re.findall(citation_pattern, result))
        assert len(output_citations) == 0, \
            f"Output should have no citations when input has none. Found: {sorted(output_citations)}"
    print("✓ Contract: Citation cleanup works with no input citations")


if __name__ == "__main__":
    print("Running Citation Cleanup Integration Tests...\n")

    tests = [
        test_phantom_removal_and_restore_work_together,
        test_citations_at_sentence_boundaries,
        test_multiple_phantom_citations_removed,
        test_valid_citations_preserved_after_phantom_removal,
        test_citation_cleanup_with_no_input_citations,
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
        print("\n🎉 All citation cleanup integration tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)

