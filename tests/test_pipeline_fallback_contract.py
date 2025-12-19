"""Contract tests for pipeline fallback paths.

These tests ensure the critical fallback logic from paragraph fusion
to sentence-by-sentence processing works correctly. If any test fails,
the fallback contract has been violated.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import process_text
from src.atlas.rhetoric import RhetoricalType
import numpy as np


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


def test_paragraph_fusion_success_no_sentence_fallback():
    """Contract: When paragraph fusion succeeds, sentence-by-sentence is NOT executed.

    This test ensures that successful paragraph fusion skips sentence processing.
    If this test fails, the fallback logic is broken.
    """
    mock_atlas = MockStyleAtlas()

    # Track if sentence-by-sentence translate was called
    sentence_translate_called = {"value": False}

    with patch('src.pipeline.StyleTranslator') as MockTranslator:
        mock_translator = MockTranslator.return_value

        # Mock translate_paragraph to return successful result (returns tuple: paragraph, rhythm_map, example)
        def mock_translate_paragraph(paragraph, atlas, author_name, style_dna=None, verbose=False, **kwargs):
            return ("Generated paragraph successfully.", None, None)
        mock_translator.translate_paragraph = mock_translate_paragraph

        # Mock translate to track if it's called (sentence-by-sentence)
        def mock_translate(*args, **kwargs):
            sentence_translate_called["value"] = True
            return "Sentence translation"
        mock_translator.translate = mock_translate

        # Mock critic to return passing result for paragraph fusion
        with patch('src.pipeline.SemanticCritic') as MockCritic:
            mock_critic = MockCritic.return_value
            mock_critic.evaluate.return_value = {
                "pass": True,
                "score": 0.85,
                "proposition_recall": 0.9,
                "style_alignment": 0.8
            }

            # Mock proposition extractor
            with patch('src.pipeline.PropositionExtractor') as MockPropExtractor:
                mock_prop_extractor = MockPropExtractor.return_value
                mock_prop_extractor.extract_atomic_propositions.return_value = [
                    "Human experience reinforces the rule of finitude",
                    "The biological cycle defines our reality"
                ]

                # Mock BlueprintExtractor to return proper SemanticBlueprint objects
                with patch('src.pipeline.BlueprintExtractor') as MockBlueprintExtractor:
                    from src.ingestion.blueprint import SemanticBlueprint
                    mock_extractor = MockBlueprintExtractor.return_value
                    def mock_extract(text, **kwargs):
                        return SemanticBlueprint(
                            original_text=text,
                            svo_triples=[],
                            named_entities=[],
                            core_keywords=set(),
                            citations=[],
                            quotes=[],
                            **kwargs
                        )
                    mock_extractor.extract = mock_extract

                    input_text = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."

                    result = process_text(
                        input_text=input_text,
                        atlas=mock_atlas,
                        author_name="Test Author",
                        style_dna="Test style DNA",
                        max_retries=1,
                        verbose=False
                    )

                    # Verify paragraph fusion succeeded
                    assert len(result) > 0, "Should return result"
                    # Verify all items are strings
                    for i, para in enumerate(result):
                        assert isinstance(para, str), f"Result[{i}] should be a string, got {type(para)}"

                    # Verify sentence-by-sentence was NOT called
                    assert not sentence_translate_called["value"], \
                        "Sentence-by-sentence translate should NOT be called when paragraph fusion succeeds"
    print("✓ Contract: Paragraph fusion success → no sentence fallback")


def test_paragraph_fusion_failure_triggers_sentence_fallback():
    """Contract: When paragraph fusion fails (low recall), sentence-by-sentence is executed.

    This test ensures the fallback to sentence-by-sentence when paragraph fusion fails.
    """
    mock_atlas = MockStyleAtlas()

    # Track if sentence-by-sentence translate was called
    sentence_translate_called = {"value": False}

    with patch('src.pipeline.StyleTranslator') as MockTranslator:
        mock_translator = MockTranslator.return_value

        # Mock translate_paragraph to return result (but will fail evaluation)
        def mock_translate_paragraph(paragraph, atlas, author_name, style_dna=None, verbose=False, **kwargs):
            return ("Generated paragraph with low recall.", None, None)
        mock_translator.translate_paragraph = mock_translate_paragraph

        # Mock translate to track if it's called (sentence-by-sentence)
        def mock_translate(*args, **kwargs):
            sentence_translate_called["value"] = True
            return "Sentence translation"
        mock_translator.translate = mock_translate

        # Mock critic to return FAILING result for paragraph fusion (low recall)
        with patch('src.pipeline.SemanticCritic') as MockCritic:
            mock_critic = MockCritic.return_value

            # First call (paragraph fusion evaluation) fails
            # Subsequent calls (sentence-by-sentence) pass
            call_count = {"value": 0}
            def mock_evaluate(*args, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    # Paragraph fusion evaluation fails
                    return {
                        "pass": False,
                        "score": 0.5,
                        "proposition_recall": 0.4,  # Below threshold
                        "style_alignment": 0.6
                    }
                else:
                    # Sentence-by-sentence evaluation passes
                    return {
                        "pass": True,
                        "score": 0.85,
                        "proposition_recall": 0.9,
                        "style_alignment": 0.8
                    }
            mock_critic.evaluate = mock_evaluate

            # Mock proposition extractor
            with patch('src.pipeline.PropositionExtractor') as MockPropExtractor:
                mock_prop_extractor = MockPropExtractor.return_value
                mock_prop_extractor.extract_atomic_propositions.return_value = [
                    "Human experience reinforces the rule of finitude",
                    "The biological cycle defines our reality"
                ]

                # Mock BlueprintExtractor to return proper SemanticBlueprint objects
                with patch('src.pipeline.BlueprintExtractor') as MockBlueprintExtractor:
                    from src.ingestion.blueprint import SemanticBlueprint
                    mock_extractor = MockBlueprintExtractor.return_value
                    def mock_extract(text, **kwargs):
                        return SemanticBlueprint(
                            original_text=text,
                            svo_triples=[],
                            named_entities=[],
                            core_keywords=set(),
                            citations=[],
                            quotes=[],
                            **kwargs
                        )
                    mock_extractor.extract = mock_extract

                    input_text = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."

                    result = process_text(
                        input_text=input_text,
                        atlas=mock_atlas,
                        author_name="Test Author",
                        style_dna="Test style DNA",
                        max_retries=1,
                        verbose=False
                    )

                    # Verify sentence-by-sentence WAS called
                    assert sentence_translate_called["value"], \
                        "Sentence-by-sentence translate SHOULD be called when paragraph fusion fails"

                    # Verify result exists
                    assert len(result) > 0, "Should return result from sentence-by-sentence fallback"
    print("✓ Contract: Paragraph fusion failure → sentence fallback executed")


def test_paragraph_fusion_exception_triggers_sentence_fallback():
    """Contract: When paragraph fusion raises exception, sentence-by-sentence is executed.

    This test ensures exception handling in paragraph fusion triggers fallback.
    """
    mock_atlas = MockStyleAtlas()

    # Track if sentence-by-sentence translate was called
    sentence_translate_called = {"value": False}

    with patch('src.pipeline.StyleTranslator') as MockTranslator:
        mock_translator = MockTranslator.return_value

        # Mock translate_paragraph to raise exception
        def mock_translate_paragraph(paragraph, atlas, author_name, style_dna=None, verbose=False, **kwargs):
            raise Exception("Paragraph fusion error")
        mock_translator.translate_paragraph = mock_translate_paragraph

        # Mock translate to track if it's called (sentence-by-sentence)
        def mock_translate(*args, **kwargs):
            sentence_translate_called["value"] = True
            return "Sentence translation"
        mock_translator.translate = mock_translate

        # Mock critic
        with patch('src.pipeline.SemanticCritic') as MockCritic:
            mock_critic = MockCritic.return_value
            mock_critic.evaluate.return_value = {
                "pass": True,
                "score": 0.85,
                "proposition_recall": 0.9,
                "style_alignment": 0.8
            }

            # Mock proposition extractor
            with patch('src.pipeline.PropositionExtractor') as MockPropExtractor:
                mock_prop_extractor = MockPropExtractor.return_value
                mock_prop_extractor.extract_atomic_propositions.return_value = [
                    "Human experience reinforces the rule of finitude",
                    "The biological cycle defines our reality"
                ]

                # Mock BlueprintExtractor to return proper SemanticBlueprint objects
                with patch('src.pipeline.BlueprintExtractor') as MockBlueprintExtractor:
                    from src.ingestion.blueprint import SemanticBlueprint
                    mock_extractor = MockBlueprintExtractor.return_value
                    def mock_extract(text, **kwargs):
                        return SemanticBlueprint(
                            original_text=text,
                            svo_triples=[],
                            named_entities=[],
                            core_keywords=set(),
                            citations=[],
                            quotes=[],
                            **kwargs
                        )
                    mock_extractor.extract = mock_extract

                    input_text = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."

                    result = process_text(
                        input_text=input_text,
                        atlas=mock_atlas,
                        author_name="Test Author",
                        style_dna="Test style DNA",
                        max_retries=1,
                        verbose=False
                    )

                    # Verify sentence-by-sentence WAS called after exception
                    assert sentence_translate_called["value"], \
                        "Sentence-by-sentence translate SHOULD be called when paragraph fusion raises exception"

                    # Verify result exists
                    assert len(result) > 0, "Should return result from sentence-by-sentence fallback"
    print("✓ Contract: Paragraph fusion exception → sentence fallback executed")


def test_multiple_paragraphs_mixed_success_failure():
    """Contract: Multiple paragraphs with mixed success/failure handled correctly.

    This test ensures the pipeline handles multiple paragraphs where some succeed
    with paragraph fusion and others fall back to sentence-by-sentence.
    """
    mock_atlas = MockStyleAtlas()

    paragraph_fusion_results = []
    sentence_fallback_results = []

    with patch('src.pipeline.StyleTranslator') as MockTranslator:
        mock_translator = MockTranslator.return_value

        # Track paragraph fusion calls
        para_fusion_call_count = {"value": 0}
        def mock_translate_paragraph(paragraph, atlas, author_name, style_dna=None, verbose=False, **kwargs):
            para_fusion_call_count["value"] += 1
            paragraph_fusion_results.append(f"Para {para_fusion_call_count['value']} fusion")
            return (f"Generated paragraph {para_fusion_call_count['value']}.", None, None)
        mock_translator.translate_paragraph = mock_translate_paragraph

        # Track sentence-by-sentence calls
        sentence_call_count = {"value": 0}
        def mock_translate(*args, **kwargs):
            sentence_call_count["value"] += 1
            sentence_fallback_results.append(f"Sentence {sentence_call_count['value']} translation")
            return f"Sentence translation {sentence_call_count['value']}."
        mock_translator.translate = mock_translate

        # Mock critic - first paragraph fails, second succeeds
        with patch('src.pipeline.SemanticCritic') as MockCritic:
            mock_critic = MockCritic.return_value

            evaluation_count = {"value": 0}
            def mock_evaluate(*args, **kwargs):
                evaluation_count["value"] += 1
                # First evaluation (para 1) fails, second (para 2) succeeds
                if evaluation_count["value"] == 1:
                    return {
                        "pass": False,
                        "score": 0.5,
                        "proposition_recall": 0.4,
                        "style_alignment": 0.6
                    }
                elif evaluation_count["value"] == 2:
                    return {
                        "pass": True,
                        "score": 0.85,
                        "proposition_recall": 0.9,
                        "style_alignment": 0.8
                    }
                else:
                    # Sentence-by-sentence evaluations
                    return {
                        "pass": True,
                        "score": 0.85,
                        "proposition_recall": 0.9,
                        "style_alignment": 0.8
                    }
            mock_critic.evaluate = mock_evaluate

            # Mock proposition extractor
            with patch('src.pipeline.PropositionExtractor') as MockPropExtractor:
                mock_prop_extractor = MockPropExtractor.return_value
                mock_prop_extractor.extract_atomic_propositions.return_value = [
                    "Human experience reinforces the rule of finitude",
                    "The biological cycle defines our reality"
                ]

                # Mock BlueprintExtractor to return proper SemanticBlueprint objects
                with patch('src.pipeline.BlueprintExtractor') as MockBlueprintExtractor:
                    from src.ingestion.blueprint import SemanticBlueprint
                    mock_extractor = MockBlueprintExtractor.return_value
                    def mock_extract(text, **kwargs):
                        return SemanticBlueprint(
                            original_text=text,
                            svo_triples=[],
                            named_entities=[],
                            core_keywords=set(),
                            citations=[],
                            quotes=[],
                            **kwargs
                        )
                    mock_extractor.extract = mock_extract

                    input_text = """First paragraph. First sentence. Second sentence.

Second paragraph. First sentence. Second sentence."""

                    result = process_text(
                        input_text=input_text,
                        atlas=mock_atlas,
                        author_name="Test Author",
                        style_dna="Test style DNA",
                        max_retries=1,
                        verbose=False
                    )

                    # Should have 2 paragraphs in result
                    assert len(result) == 2, f"Should have 2 paragraphs, got {len(result)}"

                    # First paragraph should use sentence fallback (fusion failed)
                    # Second paragraph should use fusion (succeeded)
                    assert para_fusion_call_count["value"] == 2, "Both paragraphs should attempt fusion"
                    assert sentence_call_count["value"] > 0, "First paragraph should fall back to sentences"
    print("✓ Contract: Multiple paragraphs with mixed success/failure handled correctly")


def test_context_propagation_across_fallback_boundary():
    """Contract: Context propagates correctly when falling back from paragraph to sentence.

    This test ensures context tracking works across the fallback boundary.
    """
    mock_atlas = MockStyleAtlas()

    context_tracking = []

    with patch('src.pipeline.StyleTranslator') as MockTranslator:
        mock_translator = MockTranslator.return_value

        # Mock translate_paragraph to fail
        def mock_translate_paragraph(paragraph, atlas, author_name, style_dna=None, verbose=False, **kwargs):
            return ("Generated paragraph.", None, None)
        mock_translator.translate_paragraph = mock_translate_paragraph

        # Mock translate to track context
        def mock_translate(blueprint, author_name, style_dna, rhetorical_type, examples, verbose=False):
            if blueprint.previous_context:
                context_tracking.append(blueprint.previous_context)
            return "Sentence translation."
        mock_translator.translate = mock_translate

        # Mock critic - paragraph fusion fails
        with patch('src.pipeline.SemanticCritic') as MockCritic:
            mock_critic = MockCritic.return_value
            call_count = {"value": 0}
            def mock_evaluate(*args, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 1:
                    # Paragraph fusion fails
                    return {
                        "pass": False,
                        "score": 0.5,
                        "proposition_recall": 0.4,
                        "style_alignment": 0.6
                    }
                else:
                    # Sentence evaluations pass
                    return {
                        "pass": True,
                        "score": 0.85,
                        "proposition_recall": 0.9,
                        "style_alignment": 0.8
                    }
            mock_critic.evaluate = mock_evaluate

            # Mock proposition extractor
            with patch('src.pipeline.PropositionExtractor') as MockPropExtractor:
                mock_prop_extractor = MockPropExtractor.return_value
                mock_prop_extractor.extract_atomic_propositions.return_value = [
                    "Human experience reinforces the rule of finitude",
                    "The biological cycle defines our reality"
                ]

                # Mock BlueprintExtractor to return proper SemanticBlueprint objects
                with patch('src.pipeline.BlueprintExtractor') as MockBlueprintExtractor:
                    from src.ingestion.blueprint import SemanticBlueprint
                    mock_extractor = MockBlueprintExtractor.return_value
                    def mock_extract(text, **kwargs):
                        return SemanticBlueprint(
                            original_text=text,
                            svo_triples=[],
                            named_entities=[],
                            core_keywords=set(),
                            citations=[],
                            quotes=[],
                            **kwargs
                        )
                    mock_extractor.extract = mock_extract

                    input_text = "First sentence. Second sentence. Third sentence."

                    result = process_text(
                        input_text=input_text,
                        atlas=mock_atlas,
                        author_name="Test Author",
                        style_dna="Test style DNA",
                        max_retries=1,
                        verbose=False
                    )

                    # Context should propagate between sentences in fallback
                    # First sentence has no context, second should have first sentence's context
                    assert len(context_tracking) >= 1, "Context should propagate in sentence fallback"
    print("✓ Contract: Context propagates across fallback boundary")


if __name__ == "__main__":
    print("Running Pipeline Fallback Contract Tests...\n")

    tests = [
        test_paragraph_fusion_success_no_sentence_fallback,
        test_paragraph_fusion_failure_triggers_sentence_fallback,
        test_paragraph_fusion_exception_triggers_sentence_fallback,
        test_multiple_paragraphs_mixed_success_failure,
        test_context_propagation_across_fallback_boundary,
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
        print("\n🎉 All pipeline fallback contract tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)

