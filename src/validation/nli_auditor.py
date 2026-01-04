"""NLI-based auditor for sentence-level fact verification.

Implements a "Reflexion" loop for high-accuracy text generation:
1. RECALL CHECK: Does output entail each source sentence? (catches dropped facts)
2. PRECISION CHECK: Does each output sentence contradict source? (catches hallucinations)

Uses a Cross-Encoder NLI model which is more accurate than bi-encoders for this task.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)

# NLI label mapping for cross-encoder models
LABEL_MAPPING = ['contradiction', 'entailment', 'neutral']


@dataclass
class SentenceIssue:
    """A specific sentence-level issue found by NLI audit."""
    issue_type: str  # "missing_fact" or "hallucination"
    sentence: str  # The problematic sentence
    label: str  # NLI label (contradiction, neutral, entailment)
    confidence: float  # Model confidence


@dataclass
class AuditResult:
    """Result of NLI audit on source/output pair."""
    passed: bool = True
    recall_score: float = 1.0  # Fraction of source sentences entailed
    precision_score: float = 1.0  # Fraction of output sentences not contradicting
    recall_failures: List[SentenceIssue] = field(default_factory=list)
    precision_failures: List[SentenceIssue] = field(default_factory=list)

    @property
    def all_issues(self) -> List[SentenceIssue]:
        """All issues combined."""
        return self.recall_failures + self.precision_failures

    @property
    def error_summary(self) -> str:
        """Generate error summary for repair prompt."""
        lines = []
        for issue in self.recall_failures[:3]:  # Limit to top 3
            lines.append(f"- Missing fact: '{issue.sentence[:100]}...' " if len(issue.sentence) > 100 else f"- Missing fact: '{issue.sentence}'")
        for issue in self.precision_failures[:3]:
            lines.append(f"- Contradicts source: '{issue.sentence[:100]}...'" if len(issue.sentence) > 100 else f"- Contradicts source: '{issue.sentence}'")
        return "\n".join(lines) if lines else "No specific issues identified."


class NLIAuditor:
    """Sentence-level NLI auditor for fact verification.

    Uses Cross-Encoder NLI model to check:
    - Recall: Does the output entail each source sentence?
    - Precision: Does each output sentence NOT contradict the source?
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        recall_threshold: float = 0.5,
        precision_threshold: float = 0.5,
    ):
        """Initialize the NLI auditor.

        Args:
            model_name: HuggingFace model name for NLI.
            recall_threshold: Minimum entailment probability for recall pass.
            precision_threshold: Maximum contradiction probability for precision pass.
        """
        self.model_name = model_name
        self.recall_threshold = recall_threshold
        self.precision_threshold = precision_threshold
        self._model = None
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    @property
    def model(self):
        """Lazy load NLI model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the Cross-Encoder NLI model."""
        try:
            import sys
            import warnings
            import logging
            from io import StringIO
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading NLI model: {self.model_name}")

            # Suppress output during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                transformers_logger = logging.getLogger("transformers")
                old_level = transformers_logger.level
                transformers_logger.setLevel(logging.ERROR)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                try:
                    self._model = CrossEncoder(
                        self.model_name,
                        max_length=512,
                    )
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    transformers_logger.setLevel(old_level)

            logger.info("NLI model loaded successfully")

        except ImportError:
            raise ImportError(
                "sentence-transformers is required for NLI auditing. "
                "Install with: pip install sentence-transformers"
            )

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _get_nli_scores(self, premise: str, hypothesis: str) -> Tuple[str, np.ndarray]:
        """Get NLI prediction for a premise-hypothesis pair.

        Args:
            premise: The premise text.
            hypothesis: The hypothesis text.

        Returns:
            Tuple of (predicted_label, probabilities).
        """
        # Model returns logits [contradiction, entailment, neutral]
        # Note: Some models have different ordering, but deberta-v3 uses this
        logits = self.model.predict([(premise, hypothesis)])

        if len(logits.shape) > 1:
            logits = logits[0]

        # Softmax to get probabilities
        exp_scores = np.exp(logits - np.max(logits))
        probs = exp_scores / np.sum(exp_scores)

        # Get predicted label
        pred_idx = int(np.argmax(probs))
        pred_label = LABEL_MAPPING[pred_idx]

        return pred_label, probs

    def check_recall(
        self,
        source_sentence: str,
        full_output: str,
    ) -> Tuple[bool, Optional[SentenceIssue]]:
        """Check if a source sentence is entailed by the output.

        Premise: full_output, Hypothesis: source_sentence
        "Does the output imply the source sentence is true?"

        Args:
            source_sentence: A sentence from the source text.
            full_output: The complete generated output.

        Returns:
            Tuple of (passed, issue if failed).
        """
        pred_label, probs = self._get_nli_scores(full_output, source_sentence)

        # Entailment probability (index 1 for deberta)
        entailment_prob = float(probs[1])

        if pred_label == 'entailment' or entailment_prob >= self.recall_threshold:
            return True, None

        return False, SentenceIssue(
            issue_type="missing_fact",
            sentence=source_sentence,
            label=pred_label,
            confidence=float(probs[np.argmax(probs)]),
        )

    def check_precision(
        self,
        output_sentence: str,
        full_source: str,
    ) -> Tuple[bool, Optional[SentenceIssue]]:
        """Check if an output sentence contradicts the source.

        Premise: full_source, Hypothesis: output_sentence
        "Does the output sentence contradict the source?"

        Args:
            output_sentence: A sentence from the generated output.
            full_source: The complete source text.

        Returns:
            Tuple of (passed, issue if failed).
        """
        pred_label, probs = self._get_nli_scores(full_source, output_sentence)

        # Contradiction probability (index 0 for deberta)
        contradiction_prob = float(probs[0])

        if pred_label == 'contradiction' and contradiction_prob >= self.precision_threshold:
            return False, SentenceIssue(
                issue_type="hallucination",
                sentence=output_sentence,
                label=pred_label,
                confidence=contradiction_prob,
            )

        return True, None

    def audit(self, source: str, output: str) -> AuditResult:
        """Run full NLI audit on source/output pair.

        Performs:
        1. RECALL: Check each source sentence is entailed by output
        2. PRECISION: Check each output sentence doesn't contradict source

        Args:
            source: Original source text.
            output: Generated output text.

        Returns:
            AuditResult with pass/fail and specific issues.
        """
        result = AuditResult()

        source_sentences = self.split_sentences(source)
        output_sentences = self.split_sentences(output)

        if not source_sentences or not output_sentences:
            logger.warning("Empty source or output for NLI audit")
            return result

        # 1. RECALL CHECK: Does output entail each source sentence?
        recall_passed = 0
        for source_sent in source_sentences:
            # Skip very short sentences (likely fragments)
            if len(source_sent.split()) < 3:
                recall_passed += 1
                continue

            passed, issue = self.check_recall(source_sent, output)
            if passed:
                recall_passed += 1
            elif issue:
                result.recall_failures.append(issue)

        result.recall_score = recall_passed / len(source_sentences) if source_sentences else 1.0

        # 2. PRECISION CHECK: Does each output sentence NOT contradict source?
        precision_passed = 0
        for output_sent in output_sentences:
            # Skip very short sentences
            if len(output_sent.split()) < 3:
                precision_passed += 1
                continue

            passed, issue = self.check_precision(output_sent, source)
            if passed:
                precision_passed += 1
            elif issue:
                result.precision_failures.append(issue)

        result.precision_score = precision_passed / len(output_sentences) if output_sentences else 1.0

        # Overall pass if no critical failures
        result.passed = len(result.recall_failures) == 0 and len(result.precision_failures) == 0

        logger.debug(
            f"NLI Audit: recall={result.recall_score:.2f}, "
            f"precision={result.precision_score:.2f}, "
            f"passed={result.passed}"
        )

        return result


# Singleton instance
_auditor: Optional[NLIAuditor] = None


def get_nli_auditor(
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    recall_threshold: float = 0.5,
    precision_threshold: float = 0.5,
) -> NLIAuditor:
    """Get or create singleton NLI auditor instance."""
    global _auditor
    if _auditor is None:
        _auditor = NLIAuditor(
            model_name=model_name,
            recall_threshold=recall_threshold,
            precision_threshold=precision_threshold,
        )
    return _auditor
