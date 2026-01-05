"""Tests for the NLI Auditor module.

Tests cover:
- NLIAuditor: Sentence-level NLI verification
- AuditResult: Audit result dataclass
- SentenceIssue: Issue tracking
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestAuditResult:
    """Tests for AuditResult dataclass."""

    def test_default_values(self):
        """Test default values are correct."""
        from src.validation.nli_auditor import AuditResult

        result = AuditResult()

        assert result.passed is True
        assert result.recall_score == 1.0
        assert result.precision_score == 1.0
        assert len(result.recall_failures) == 0
        assert len(result.precision_failures) == 0

    def test_all_issues_combines_failures(self):
        """Test all_issues combines recall and precision failures."""
        from src.validation.nli_auditor import AuditResult, SentenceIssue

        result = AuditResult()
        result.recall_failures = [
            SentenceIssue("missing_fact", "Sentence 1", "neutral", 0.8)
        ]
        result.precision_failures = [
            SentenceIssue("hallucination", "Sentence 2", "contradiction", 0.9)
        ]

        assert len(result.all_issues) == 2
        assert result.all_issues[0].issue_type == "missing_fact"
        assert result.all_issues[1].issue_type == "hallucination"

    def test_error_summary_formats_issues(self):
        """Test error_summary generates readable format."""
        from src.validation.nli_auditor import AuditResult, SentenceIssue

        result = AuditResult()
        result.recall_failures = [
            SentenceIssue("missing_fact", "The IMF issued a warning.", "neutral", 0.8)
        ]
        result.precision_failures = [
            SentenceIssue("hallucination", "The world ended in 2020.", "contradiction", 0.9)
        ]

        summary = result.error_summary

        assert "Missing fact" in summary
        assert "Contradicts source" in summary
        assert "IMF" in summary

    def test_error_summary_empty_when_no_issues(self):
        """Test error_summary handles no issues."""
        from src.validation.nli_auditor import AuditResult

        result = AuditResult()

        assert "No specific issues" in result.error_summary


class TestSentenceIssue:
    """Tests for SentenceIssue dataclass."""

    def test_creation(self):
        """Test SentenceIssue creation."""
        from src.validation.nli_auditor import SentenceIssue

        issue = SentenceIssue(
            issue_type="missing_fact",
            sentence="Test sentence.",
            label="neutral",
            confidence=0.85,
        )

        assert issue.issue_type == "missing_fact"
        assert issue.sentence == "Test sentence."
        assert issue.label == "neutral"
        assert issue.confidence == 0.85


class TestNLIAuditor:
    """Tests for NLIAuditor class."""

    @pytest.fixture
    def mock_nlp(self):
        """Create mock spaCy nlp."""
        mock = MagicMock()

        # Create mock sentences
        mock_sent1 = MagicMock()
        mock_sent1.text = "First sentence here."
        mock_sent2 = MagicMock()
        mock_sent2.text = "Second sentence here."

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent1, mock_sent2]
        mock.return_value = mock_doc

        return mock

    @pytest.fixture
    def mock_cross_encoder(self):
        """Create mock CrossEncoder."""
        mock_model = MagicMock()
        # Return logits [contradiction, entailment, neutral]
        # High entailment score
        mock_model.predict.return_value = np.array([-2.0, 3.0, -1.0])
        return mock_model

    def test_init_default_values(self):
        """Test NLIAuditor initialization with defaults."""
        from src.validation.nli_auditor import NLIAuditor

        auditor = NLIAuditor()

        assert auditor.model_name == "cross-encoder/nli-deberta-v3-base"
        assert auditor.recall_threshold == 0.5
        assert auditor.precision_threshold == 0.5

    def test_init_custom_values(self):
        """Test NLIAuditor initialization with custom values."""
        from src.validation.nli_auditor import NLIAuditor

        auditor = NLIAuditor(
            model_name="custom-model",
            recall_threshold=0.7,
            precision_threshold=0.6,
        )

        assert auditor.model_name == "custom-model"
        assert auditor.recall_threshold == 0.7
        assert auditor.precision_threshold == 0.6

    @patch('src.validation.nli_auditor.get_nlp')
    def test_split_sentences(self, mock_get_nlp, mock_nlp):
        """Test sentence splitting."""
        from src.validation.nli_auditor import NLIAuditor

        mock_get_nlp.return_value = mock_nlp

        auditor = NLIAuditor()
        sentences = auditor.split_sentences("First sentence. Second sentence.")

        assert len(sentences) == 2
        assert "First" in sentences[0]
        assert "Second" in sentences[1]

    @patch('src.validation.nli_auditor.get_nlp')
    def test_check_recall_passes_on_entailment(self, mock_get_nlp, mock_nlp, mock_cross_encoder):
        """Test check_recall passes when output entails source sentence."""
        from src.validation.nli_auditor import NLIAuditor

        mock_get_nlp.return_value = mock_nlp

        auditor = NLIAuditor()
        auditor._model = mock_cross_encoder

        passed, issue = auditor.check_recall(
            "The IMF warned about risks.",
            "The IMF issued a warning about risks in the economy."
        )

        assert passed is True
        assert issue is None

    @patch('src.validation.nli_auditor.get_nlp')
    def test_check_recall_fails_on_neutral(self, mock_get_nlp, mock_nlp):
        """Test check_recall fails when output doesn't entail source sentence."""
        from src.validation.nli_auditor import NLIAuditor

        mock_get_nlp.return_value = mock_nlp

        # Create mock model returning neutral
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-1.0, -2.0, 3.0])  # High neutral

        auditor = NLIAuditor()
        auditor._model = mock_model

        passed, issue = auditor.check_recall(
            "The IMF warned about risks.",
            "The weather was nice today."
        )

        assert passed is False
        assert issue is not None
        assert issue.issue_type == "missing_fact"

    @patch('src.validation.nli_auditor.get_nlp')
    def test_check_precision_passes_on_non_contradiction(self, mock_get_nlp, mock_nlp, mock_cross_encoder):
        """Test check_precision passes when output doesn't contradict source."""
        from src.validation.nli_auditor import NLIAuditor

        mock_get_nlp.return_value = mock_nlp

        auditor = NLIAuditor()
        auditor._model = mock_cross_encoder

        passed, issue = auditor.check_precision(
            "The economy grew by 3%.",
            "The economy showed positive growth this year."
        )

        assert passed is True
        assert issue is None

    @patch('src.validation.nli_auditor.get_nlp')
    def test_check_precision_fails_on_contradiction(self, mock_get_nlp, mock_nlp):
        """Test check_precision fails when output contradicts source."""
        from src.validation.nli_auditor import NLIAuditor

        mock_get_nlp.return_value = mock_nlp

        # Create mock model returning contradiction
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.0, -2.0, -1.0])  # High contradiction

        auditor = NLIAuditor()
        auditor._model = mock_model

        passed, issue = auditor.check_precision(
            "The world ended in 2020.",
            "The economy grew by 3%."
        )

        assert passed is False
        assert issue is not None
        assert issue.issue_type == "hallucination"

    @patch('src.validation.nli_auditor.get_nlp')
    def test_audit_all_pass(self, mock_get_nlp):
        """Test audit returns passed when all checks pass."""
        from src.validation.nli_auditor import NLIAuditor

        # Setup mock nlp
        mock_nlp = MagicMock()
        mock_sent = MagicMock()
        mock_sent.text = "Test sentence here."
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        mock_nlp.return_value = mock_doc
        mock_get_nlp.return_value = mock_nlp

        # Setup mock model - all entailment
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-2.0, 3.0, -1.0])

        auditor = NLIAuditor()
        auditor._model = mock_model

        result = auditor.audit(
            "The IMF issued a warning.",
            "The IMF warned about economic risks."
        )

        assert result.passed is True
        assert result.recall_score == 1.0
        assert result.precision_score == 1.0

    @patch('src.validation.nli_auditor.get_nlp')
    def test_audit_recall_failure(self, mock_get_nlp):
        """Test audit detects recall failures."""
        from src.validation.nli_auditor import NLIAuditor

        # Setup mock nlp with multiple sentences
        mock_nlp = MagicMock()
        mock_sent1 = MagicMock()
        mock_sent1.text = "First important fact here."
        mock_sent2 = MagicMock()
        mock_sent2.text = "Second important fact here."
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent1, mock_sent2]
        mock_nlp.return_value = mock_doc
        mock_get_nlp.return_value = mock_nlp

        # Setup mock model - returns neutral (not entailed)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-1.0, -2.0, 3.0])  # High neutral

        auditor = NLIAuditor()
        auditor._model = mock_model

        result = auditor.audit(
            "First fact. Second fact.",
            "Completely different content."
        )

        assert result.passed is False
        assert result.recall_score < 1.0
        assert len(result.recall_failures) > 0


class TestGetNLIAuditor:
    """Tests for get_nli_auditor factory function."""

    def test_returns_auditor(self):
        """Test get_nli_auditor returns an NLIAuditor instance."""
        from src.validation.nli_auditor import get_nli_auditor, NLIAuditor

        # Reset singleton
        import src.validation.nli_auditor as module
        module._auditor = None

        auditor = get_nli_auditor()

        assert isinstance(auditor, NLIAuditor)

    def test_returns_same_instance(self):
        """Test get_nli_auditor returns singleton."""
        from src.validation.nli_auditor import get_nli_auditor

        # Reset singleton
        import src.validation.nli_auditor as module
        module._auditor = None

        auditor1 = get_nli_auditor()
        auditor2 = get_nli_auditor()

        assert auditor1 is auditor2


class TestNLIAuditorIntegration:
    """Integration tests for NLI Auditor with transfer pipeline."""

    def test_transfer_config_has_nli_settings(self):
        """Test TransferConfig includes NLI settings."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()

        assert hasattr(config, 'use_sentence_nli')
        assert hasattr(config, 'nli_model')
        assert hasattr(config, 'nli_recall_threshold')
        assert hasattr(config, 'nli_precision_threshold')

    def test_transfer_config_nli_defaults(self):
        """Test TransferConfig NLI default values."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()

        assert config.use_sentence_nli is False
        assert config.nli_model == "cross-encoder/nli-deberta-v3-base"
        assert config.nli_recall_threshold == 0.5
        assert config.nli_precision_threshold == 0.5

    def test_transfer_stats_has_nli_fields(self):
        """Test TransferStats includes NLI tracking fields."""
        from src.generation.transfer import TransferStats

        stats = TransferStats()

        assert hasattr(stats, 'nli_recall_scores')
        assert hasattr(stats, 'nli_precision_scores')
        assert hasattr(stats, 'nli_repairs_made')

    def test_transfer_stats_to_dict_includes_nli(self):
        """Test TransferStats.to_dict includes NLI stats when available."""
        from src.generation.transfer import TransferStats

        stats = TransferStats()
        stats.paragraphs_processed = 1
        stats.nli_recall_scores = [0.9, 0.8]
        stats.nli_precision_scores = [0.95, 0.85]
        stats.nli_repairs_made = 1

        result = stats.to_dict()

        assert "avg_nli_recall" in result
        assert "avg_nli_precision" in result
        assert "nli_repairs_made" in result
        assert result["avg_nli_recall"] == 0.85
        assert result["avg_nli_precision"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
