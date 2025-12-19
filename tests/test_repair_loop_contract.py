"""Contract tests for repair loop in translate_paragraph.

These tests ensure the repair loop contract is maintained: full checklist format,
generation failure handling, flatline detection, and missing proposition identification.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.translator import StyleTranslator


class MockLLMProvider:
    """Mock LLM provider for testing repair loop."""

    def __init__(self, repair_responses=None):
        self.call_count = 0
        self.call_history = []
        self.repair_responses = repair_responses or []

    def call(self, system_prompt, user_prompt, model_type="editor", require_json=False, temperature=0.7, max_tokens=500, timeout=None, top_p=None):
        self.call_count += 1
        self.call_history.append({
            "system_prompt": system_prompt[:100] if system_prompt else "",
            "user_prompt": user_prompt[:200] if user_prompt else "",
            "call_number": self.call_count
        })

        # Check if this is a repair call
        if "missed these specific facts" in user_prompt or "FULL CHECKLIST" in user_prompt:
            if self.repair_responses and len(self.repair_responses) >= self.call_count:
                return self.repair_responses[self.call_count - 1]
            # Default repair response
            return json.dumps([
                "It is through the dialectical process that human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality."
            ])
        elif "extract every distinct fact" in user_prompt.lower():
            return json.dumps([
                "Human experience reinforces the rule of finitude",
                "The biological cycle of birth, life, and decay defines our reality",
                "Every object we touch eventually breaks"
            ])
        else:
            # Initial generation
            return json.dumps([
                "Human experience reinforces the rule of finitude."
            ])


class MockStyleAtlas:
    """Mock StyleAtlas for testing."""

    def __init__(self):
        self.examples = [
            "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction."
        ]
        self.author_style_vector = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)

    def get_examples_by_rhetoric(self, rhetorical_type, top_k, author_name, query_text=None, exclude=None):
        return self.examples[:top_k]

    def get_author_style_vector(self, author_name: str):
        return self.author_style_vector


def test_repair_loop_full_checklist_format():
    """Contract: Repair prompt includes FULL CHECKLIST of all propositions.

    This test ensures the repair prompt includes both preserved and missing propositions.
    If this test fails, the full checklist fix is broken.
    """
    translator = StyleTranslator()

    # Mock LLM to track repair prompt content
    repair_prompts = []
    def mock_call(system_prompt, user_prompt, **kwargs):
        if "missed these specific facts" in user_prompt or "FULL CHECKLIST" in user_prompt:
            repair_prompts.append(user_prompt)
        return json.dumps([
            "Human experience reinforces the rule of finitude. The biological cycle defines our reality."
        ])

    translator.llm_provider = Mock()
    translator.llm_provider.call = mock_call

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.8,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Mock critic to return low recall (triggers repair)
    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value

        call_count = {"value": 0}
        def mock_evaluate(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                # Initial evaluation: low recall
                return {
                    "pass": False,
                    "score": 0.6,
                    "proposition_recall": 0.5,  # Below threshold
                    "style_alignment": 0.7,
                    "feedback": "Missing propositions",
                    "recall_details": {
                        "preserved": ["Human experience reinforces the rule of finitude"],
                        "missing": [
                            "The biological cycle of birth, life, and decay defines our reality",
                            "Every object we touch eventually breaks"
                        ],
                        "scores": {}
                    }
                }
            else:
                # Repair evaluation: improved
                return {
                    "pass": True,
                    "score": 0.85,
                    "proposition_recall": 0.9,
                    "style_alignment": 0.8,
                    "feedback": "Passed",
                    "recall_details": {
                        "preserved": [
                            "Human experience reinforces the rule of finitude",
                            "The biological cycle of birth, life, and decay defines our reality",
                            "Every object we touch eventually breaks"
                        ],
                        "missing": [],
                        "scores": {}
                    }
                }
        mock_critic_instance.evaluate = mock_evaluate

        input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks."

        result = translator.translate_paragraph(
            paragraph=input_paragraph,
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=False
        )

        # Verify repair prompt was called
        assert len(repair_prompts) > 0, "Repair prompt should be called"

        # Verify FULL CHECKLIST format
        repair_prompt = repair_prompts[0]
        assert "FULL CHECKLIST" in repair_prompt or "all propositions" in repair_prompt.lower(), \
            "Repair prompt should include FULL CHECKLIST"

        # Verify both preserved and missing propositions are listed
        assert "Human experience reinforces the rule of finitude" in repair_prompt, \
            "Preserved proposition should be in checklist"
        assert "The biological cycle" in repair_prompt or "biological cycle" in repair_prompt, \
            "Missing proposition should be in checklist"
    print("✓ Contract: Repair prompt includes FULL CHECKLIST")


def test_repair_loop_generation_failure_stops_gracefully():
    """Contract: Repair generation failure stops loop gracefully.

    This test ensures the repair loop doesn't crash when generation fails.
    """
    translator = StyleTranslator()

    # Mock LLM to fail on repair call
    call_count = {"value": 0}
    def mock_call(system_prompt, user_prompt, **kwargs):
        call_count["value"] += 1
        if "missed these specific facts" in user_prompt or "FULL CHECKLIST" in user_prompt:
            raise Exception("Repair generation failed")
        return json.dumps([
            "Human experience reinforces the rule of finitude."
        ])

    translator.llm_provider = Mock()
    translator.llm_provider.call = mock_call

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.8,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Mock critic to return low recall (triggers repair)
    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value
        mock_critic_instance.evaluate.return_value = {
            "pass": False,
            "score": 0.6,
            "proposition_recall": 0.5,
            "style_alignment": 0.7,
            "feedback": "Missing propositions",
            "recall_details": {
                "preserved": ["Human experience reinforces the rule of finitude"],
                "missing": ["The biological cycle defines our reality"],
                "scores": {}
            }
        }

        input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle defines our reality."

        # Should not crash, should return best available
        result = translator.translate_paragraph(
            paragraph=input_paragraph,
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=False
        )

        # Should return something (either original or best candidate)
        assert result is not None, "Should return result even when repair generation fails"
        assert len(result) > 0, "Result should not be empty"
    print("✓ Contract: Repair generation failure → loop stops gracefully")


def test_repair_loop_flatline_stops_after_max_attempts():
    """Contract: Repair loop stops after max attempts when recall doesn't improve.

    This test ensures the loop doesn't run indefinitely when recall flatlines.
    """
    translator = StyleTranslator()

    # Mock LLM to return same low-quality repair
    repair_count = {"value": 0}
    def mock_call(system_prompt, user_prompt, **kwargs):
        if "missed these specific facts" in user_prompt or "FULL CHECKLIST" in user_prompt:
            repair_count["value"] += 1
            # Return repair that still has low recall
            return json.dumps([
                "Human experience reinforces the rule of finitude."  # Still missing propositions
            ])
        return json.dumps([
            "Human experience reinforces the rule of finitude."
        ])

    translator.llm_provider = Mock()
    translator.llm_provider.call = mock_call

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.8,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Mock critic to always return low recall (flatline)
    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value
        mock_critic_instance.evaluate.return_value = {
            "pass": False,
            "score": 0.5,
            "proposition_recall": 0.4,  # Always below threshold
            "style_alignment": 0.6,
            "feedback": "Missing propositions",
            "recall_details": {
                "preserved": ["Human experience reinforces the rule of finitude"],
                "missing": ["The biological cycle defines our reality"],
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

        # Should stop after max attempts (2)
        assert repair_count["value"] <= 2, \
            f"Repair loop should stop after max attempts (2), but ran {repair_count['value']} times"

        # Should return best available
        assert result is not None, "Should return result even when repair flatlines"
    print("✓ Contract: Repair flatline → stops after max attempts")


def test_repair_loop_missing_propositions_identified_correctly():
    """Contract: Missing propositions are correctly identified from recall_details.

    This test ensures the repair loop correctly extracts missing propositions.
    """
    translator = StyleTranslator()

    missing_props_tracked = []
    def mock_call(system_prompt, user_prompt, **kwargs):
        if "missed these specific facts" in user_prompt or "FULL CHECKLIST" in user_prompt:
            # Track what missing propositions were identified
            if "biological cycle" in user_prompt:
                missing_props_tracked.append("biological cycle")
            if "object we touch" in user_prompt:
                missing_props_tracked.append("object we touch")
        return json.dumps([
            "Human experience reinforces the rule of finitude. The biological cycle defines our reality. Every object we touch eventually breaks."
        ])

    translator.llm_provider = Mock()
    translator.llm_provider.call = mock_call

    translator.paragraph_fusion_config = {
        "num_style_examples": 2,
        "num_variations": 1,
        "proposition_recall_threshold": 0.8,
        "min_word_count": 20,
        "min_sentence_count": 1,
        "retrieval_pool_size": 5
    }

    mock_atlas = MockStyleAtlas()

    # Mock critic with specific missing propositions
    with patch('src.validator.semantic_critic.SemanticCritic') as MockCritic:
        mock_critic_instance = MockCritic.return_value

        call_count = {"value": 0}
        def mock_evaluate(*args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return {
                    "pass": False,
                    "score": 0.6,
                    "proposition_recall": 0.33,  # Only 1 of 3 propositions
                    "style_alignment": 0.7,
                    "feedback": "Missing propositions",
                    "recall_details": {
                        "preserved": ["Human experience reinforces the rule of finitude"],
                        "missing": [
                            "The biological cycle of birth, life, and decay defines our reality",
                            "Every object we touch eventually breaks"
                        ],
                        "scores": {}
                    }
                }
            else:
                return {
                    "pass": True,
                    "score": 0.9,
                    "proposition_recall": 1.0,
                    "style_alignment": 0.8,
                    "feedback": "Passed",
                    "recall_details": {
                        "preserved": [
                            "Human experience reinforces the rule of finitude",
                            "The biological cycle of birth, life, and decay defines our reality",
                            "Every object we touch eventually breaks"
                        ],
                        "missing": [],
                        "scores": {}
                    }
                }
        mock_critic_instance.evaluate = mock_evaluate

        input_paragraph = "Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks."

        result = translator.translate_paragraph(
            paragraph=input_paragraph,
            atlas=mock_atlas,
            author_name="Test Author",
            verbose=False
        )

        # Verify missing propositions were identified
        assert len(missing_props_tracked) > 0, "Missing propositions should be identified"
    print("✓ Contract: Missing propositions identified correctly")


if __name__ == "__main__":
    print("Running Repair Loop Contract Tests...\n")

    tests = [
        test_repair_loop_full_checklist_format,
        test_repair_loop_generation_failure_stops_gracefully,
        test_repair_loop_flatline_stops_after_max_attempts,
        test_repair_loop_missing_propositions_identified_correctly,
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
        print("\n🎉 All repair loop contract tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)

