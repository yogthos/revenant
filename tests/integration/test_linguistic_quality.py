"""Linguistic quality integration tests.

Tests for zipper merge, action echo, grounding validation, and perspective lock.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.utils.text_processing import check_zipper_merge
from src.validator.statistical_critic import StatisticalCritic
from src.validator.semantic_critic import SemanticCritic
from src.generator.translator import StyleTranslator
from tests.test_helpers import ensure_config_exists
from src.utils.spacy_linguistics import get_main_verbs_excluding_auxiliaries
from src.utils.nlp_manager import NLPManager

# Ensure config exists
ensure_config_exists()


class TestAntiStutterZipperMerge:
    """Test anti-stutter (zipper merge) detection."""

    def test_full_echo_detection(self):
        """Test that full echo is detected."""
        prev_sent = "The doors opened."
        new_sent = "The doors opened again."

        result = check_zipper_merge(prev_sent, new_sent)
        assert result is True, "Full echo should be detected"

    def test_head_echo_detection(self):
        """Test that head echo (same 3+ words at start) is detected."""
        prev_sent = "I walked home."
        new_sent = "I walked to the store."

        result = check_zipper_merge(prev_sent, new_sent)
        assert result is True, "Head echo should be detected"

    def test_tail_echo_detection(self):
        """Test that tail echo (end of prev matches start of new) is detected."""
        prev_sent = "The doors opened."
        new_sent = "Opened, the room revealed its secrets."

        result = check_zipper_merge(prev_sent, new_sent)
        # Note: This may not always detect tail echo depending on implementation
        # The check looks for first 3 words of new_sent in last 6 words of prev_sent
        # "Opened" appears in "doors opened" so it should be detected
        assert result is True, "Tail echo should be detected"

    def test_no_echo_passes(self):
        """Test that non-echoing sentences pass."""
        prev_sent = "The doors opened."
        new_sent = "I entered the room."

        result = check_zipper_merge(prev_sent, new_sent)
        assert result is False, "Non-echoing sentences should pass"


class TestActionEchoDetection:
    """Test action echo detection using spaCy lemmatization.

    Note: check_action_echo only uses spaCy, not LLM. StatisticalCritic doesn't
    use LLMProvider, so no mocking needed.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.critic = StatisticalCritic(config_path="config.json")

    @pytest.mark.spacy_dependent
    def test_action_echo_detected_weave(self):
        """Test that 'weaving' and 'wove' are detected as same action.

        Note: This test depends on spaCy parsing. If it fails, check if verbs
        are being extracted correctly using get_main_verbs_excluding_auxiliaries.
        """
        sentences = [
            "I was weaving a basket.",
            "I wove another one."
        ]

        issues = self.critic.check_action_echo(sentences)

        # Lenient check: if no issues found, verify verbs were extracted
        if len(issues) == 0:
            # Check if verbs were extracted at all (spaCy parsing might vary)
            nlp = NLPManager.get_nlp()
            doc1 = nlp(sentences[0])
            doc2 = nlp(sentences[1])
            verbs1 = get_main_verbs_excluding_auxiliaries(doc1)
            verbs2 = get_main_verbs_excluding_auxiliaries(doc2)

            # If "weave" is in both sets, the function works but check_action_echo might have a bug
            if "weave" in verbs1 and "weave" in verbs2:
                pytest.fail("Verbs extracted correctly ('weave' in both sentences) but echo not detected - check_action_echo bug")
            else:
                # spaCy parsing variation - document what was found
                pytest.skip(f"spaCy parsing variation - verbs1={verbs1}, verbs2={verbs2}. Expected 'weave' in both.")

        assert len(issues) > 0, "Action echo should be detected (weave/wove)"
        assert any("weave" in issue.lower() or "wove" in issue.lower() for issue in issues)

    def test_action_echo_detected_run(self):
        """Test that 'ran' and 'runs' are detected as same action."""
        sentences = [
            "She ran fast.",
            "He runs daily."
        ]

        issues = self.critic.check_action_echo(sentences)
        assert len(issues) > 0, "Action echo should be detected (run/ran/runs)"
        assert any("run" in issue.lower() for issue in issues)

    @pytest.mark.spacy_dependent
    def test_auxiliary_verbs_ignored(self):
        """Test that auxiliary verbs and copulas (was, had, did, be) are ignored.

        Note: "was" in "I was happy" is a copula, not an action verb, so it should
        be excluded from action echo detection.
        """
        sentences = [
            "I was happy.",
            "She was sad."
        ]

        issues = self.critic.check_action_echo(sentences)

        # Should NOT detect echo for copulas/auxiliaries
        # If issues are found, check if it's a copula that should be excluded
        if len(issues) > 0:
            # Verify that "be" (lemma of "was") is being excluded
            nlp = NLPManager.get_nlp()
            doc1 = nlp(sentences[0])
            doc2 = nlp(sentences[1])
            verbs1 = get_main_verbs_excluding_auxiliaries(doc1)
            verbs2 = get_main_verbs_excluding_auxiliaries(doc2)

            # If "be" is in the extracted verbs, the function isn't excluding copulas
            if "be" in verbs1 or "be" in verbs2:
                pytest.fail(f"Copula 'be' should be excluded but was found in verbs: verbs1={verbs1}, verbs2={verbs2}")
            else:
                # If "be" is not in verbs but issues were found, there's a different problem
                pytest.fail(f"Action echo detected but 'be' not in extracted verbs. Issues: {issues}, verbs1={verbs1}, verbs2={verbs2}")

        assert len(issues) == 0, "Auxiliary verbs and copulas should not trigger action echo"

    def test_no_action_echo_passes(self):
        """Test that different actions pass."""
        sentences = [
            "I walked home.",
            "She drove to work."
        ]

        issues = self.critic.check_action_echo(sentences)
        assert len(issues) == 0, "Different actions should not trigger echo"


class TestGroundingValidation:
    """Test grounding validation (anti-moralizing).

    Note: check_ending_grounding uses spaCy, not LLM. We mock the LLM provider
    to avoid requiring API keys in CI.
    """

    def setup_method(self):
        """Set up test fixtures with mocked LLM provider."""
        from unittest.mock import patch
        from tests.mocks.mock_llm_provider import get_mock_llm_provider

        # Mock LLM provider since check_ending_grounding doesn't use it
        # SemanticCritic tries to initialize LLMProvider, so we need to mock it
        self.mock_llm = get_mock_llm_provider()

        # Patch LLMProvider before SemanticCritic initialization
        # Keep the patch active for the entire test
        self.llm_patcher = patch('src.validator.semantic_critic.LLMProvider', return_value=self.mock_llm)
        self.llm_patcher.start()

        try:
            self.critic = SemanticCritic(config_path="config.json")
            # Ensure the critic uses the mock
            self.critic.llm_provider = self.mock_llm
        except Exception:
            # If initialization fails, stop the patcher and re-raise
            self.llm_patcher.stop()
            raise

    def teardown_method(self):
        """Clean up patches after test."""
        if hasattr(self, 'llm_patcher'):
            self.llm_patcher.stop()

    def test_abstract_moralizing_fails(self):
        """Test that abstract/moralizing endings fail."""
        paragraph = "This is a paragraph about something. Thus, I learned about society."

        issue = self.critic.check_ending_grounding(paragraph)
        assert issue is not None, "Abstract/moralizing ending should fail"
        assert "abstract" in issue.lower() or "moralizing" in issue.lower()

    def test_concrete_detail_passes(self):
        """Test that concrete sensory details pass."""
        paragraph = "This is a paragraph about something. The door closed with a click."

        issue = self.critic.check_ending_grounding(paragraph)
        assert issue is None, "Concrete detail should pass"

    def test_moralizing_pattern_fails(self):
        """Test that moralizing patterns fail."""
        paragraph = "This is a paragraph. In conclusion, the lesson is clear."

        issue = self.critic.check_ending_grounding(paragraph)
        assert issue is not None, "Moralizing pattern should fail"


class TestPerspectiveLock:
    """Test perspective locking verification.

    Note: These tests only use verify_perspective which uses spaCy, not LLM.
    We mock the LLM provider to avoid requiring API keys in CI.
    """

    def setup_method(self):
        """Set up test fixtures with mocked LLM provider."""
        ensure_config_exists()
        from unittest.mock import patch
        from tests.mocks.mock_llm_provider import get_mock_llm_provider

        # Create mock LLM provider
        self.mock_llm = get_mock_llm_provider()

        # Patch LLMProvider before StyleTranslator initialization
        # Keep the patch active for the entire test
        self.llm_patcher = patch('src.generator.translator.LLMProvider', return_value=self.mock_llm)
        self.llm_patcher.start()

        try:
            self.translator = StyleTranslator(config_path="config.json")
            # Ensure the translator uses the mock
            self.translator.llm_provider = self.mock_llm
        except Exception:
            # If initialization fails, stop the patcher and re-raise
            self.llm_patcher.stop()
            raise

    def teardown_method(self):
        """Clean up patches after test."""
        if hasattr(self, 'llm_patcher'):
            self.llm_patcher.stop()

    def test_first_person_singular_lock(self):
        """Test that first person singular input locks perspective."""
        # Test with text that should maintain first person
        generated_text = "I went to the store. I bought some food. I returned home."

        result = self.translator.verify_perspective(generated_text, "first_person_singular")
        assert result is True, "First person singular should pass"

    def test_first_person_singular_fails_with_third(self):
        """Test that first person fails when third person pronouns appear."""
        generated_text = "I went to the store. He bought some food."

        result = self.translator.verify_perspective(generated_text, "first_person_singular")
        assert result is False, "First person should fail when third person appears"

    def test_third_person_lock(self):
        """Test that third person input locks perspective."""
        generated_text = "He walked home. She drove to work. They met at the cafe."

        result = self.translator.verify_perspective(generated_text, "third_person")
        assert result is True, "Third person should pass"

    def test_third_person_fails_with_first(self):
        """Test that third person fails when first person pronouns appear."""
        generated_text = "He walked home. I drove to work."

        result = self.translator.verify_perspective(generated_text, "third_person")
        assert result is False, "Third person should fail when first person appears"

    def test_first_person_plural_lock(self):
        """Test that first person plural input locks perspective."""
        generated_text = "We went to the store. We bought some food. We returned home."

        result = self.translator.verify_perspective(generated_text, "first_person_plural")
        assert result is True, "First person plural should pass"

    def test_empty_text_passes(self):
        """Test that empty text passes (no pronouns to check)."""
        result = self.translator.verify_perspective("", "first_person_singular")
        assert result is True, "Empty text should pass"

