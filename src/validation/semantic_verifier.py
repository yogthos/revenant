"""Robust multi-layer semantic verification for style transfer.

This module provides comprehensive semantic verification that catches:
1. Sentence-level hallucinations (output sentences not grounded in source)
2. Content loss (source information missing from output)
3. Named entity fabrication (entities in output not in source)
4. Topic drift (output discussing different subject matter)

The key insight is that paragraph-level NLI is insufficient - we need
sentence-level verification to catch localized hallucinations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Lazy-loaded models
_nli_model = None
_nlp = None


def _get_nli_model():
    """Get the NLI model, loading if necessary."""
    global _nli_model
    if _nli_model is None:
        try:
            import sys
            import warnings
            import logging
            from io import StringIO
            from sentence_transformers import CrossEncoder

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                transformers_logger = logging.getLogger("transformers")
                old_level = transformers_logger.level
                transformers_logger.setLevel(logging.ERROR)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                try:
                    _nli_model = CrossEncoder(
                        "cross-encoder/nli-deberta-v3-small",
                        max_length=512,
                    )
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    transformers_logger.setLevel(old_level)
            logger.info("Loaded NLI model for semantic verification")
        except ImportError:
            logger.warning("sentence-transformers not available for NLI")
            _nli_model = None
    return _nli_model


def _get_nlp():
    """Get spaCy model for NLP processing."""
    global _nlp
    if _nlp is None:
        from ..utils.nlp import get_nlp
        _nlp = get_nlp()
    return _nlp


@dataclass
class SentenceVerification:
    """Verification result for a single output sentence."""
    output_sentence: str
    best_source_match: Optional[str]
    entailment_score: float
    is_grounded: bool  # True if sentence is grounded in source
    is_hallucination: bool  # True if sentence appears fabricated


@dataclass
class VerificationResult:
    """Comprehensive verification result."""
    # Overall scores
    overall_score: float  # Combined score (0-1)
    is_valid: bool  # Meets threshold

    # Sentence-level analysis
    sentence_scores: List[SentenceVerification] = field(default_factory=list)
    grounded_ratio: float = 0.0  # Fraction of output sentences grounded in source
    hallucination_count: int = 0  # Number of likely hallucinated sentences

    # Content coverage
    content_word_coverage: float = 0.0  # Fraction of source content words in output
    missing_content_words: List[str] = field(default_factory=list)

    # Named entity analysis
    entity_coverage: float = 0.0  # Fraction of source entities in output
    missing_entities: List[str] = field(default_factory=list)
    fabricated_entities: List[str] = field(default_factory=list)

    # Diagnostics
    forward_entailment: float = 0.0  # source → output
    backward_entailment: float = 0.0  # output → source

    def get_issues(self) -> List[str]:
        """Get human-readable list of issues found."""
        issues = []
        if self.hallucination_count > 0:
            issues.append(f"{self.hallucination_count} hallucinated sentence(s)")
        if self.grounded_ratio < 0.7:
            issues.append(f"Only {self.grounded_ratio:.0%} of output grounded in source")
        if self.missing_entities:
            issues.append(f"Missing entities: {', '.join(self.missing_entities[:3])}")
        if self.fabricated_entities:
            issues.append(f"Fabricated entities: {', '.join(self.fabricated_entities[:3])}")
        if self.content_word_coverage < 0.5:
            issues.append(f"Low content coverage: {self.content_word_coverage:.0%}")
        return issues


class SemanticVerifier:
    """Multi-layer semantic verification for style transfer outputs.

    Verification layers:
    1. Sentence-level grounding: Each output sentence must be entailed by some source sentence
    2. Content word coverage: Key content words from source should appear in output
    3. Named entity verification: Entities in output must exist in source
    4. Bidirectional entailment: Overall semantic equivalence check
    """

    def __init__(
        self,
        entailment_threshold: float = 0.5,
        grounding_threshold: float = 0.4,
        content_coverage_weight: float = 0.3,
        entity_coverage_weight: float = 0.3,
        grounding_weight: float = 0.4,
    ):
        """Initialize the verifier.

        Args:
            entailment_threshold: Min entailment score for sentence grounding.
            grounding_threshold: Min score to consider a sentence grounded.
            content_coverage_weight: Weight for content word coverage in final score.
            entity_coverage_weight: Weight for entity coverage in final score.
            grounding_weight: Weight for sentence grounding in final score.
        """
        self.entailment_threshold = entailment_threshold
        self.grounding_threshold = grounding_threshold
        self.content_coverage_weight = content_coverage_weight
        self.entity_coverage_weight = entity_coverage_weight
        self.grounding_weight = grounding_weight

    def verify(
        self,
        source: str,
        output: str,
        threshold: float = 0.7,
    ) -> VerificationResult:
        """Perform comprehensive semantic verification.

        Args:
            source: Original source text.
            output: Generated output text.
            threshold: Overall score threshold for validity.

        Returns:
            VerificationResult with detailed analysis.
        """
        nlp = _get_nlp()
        nli_model = _get_nli_model()

        # Split into sentences
        source_sents = self._split_sentences(source)
        output_sents = self._split_sentences(output)

        if not source_sents or not output_sents:
            return VerificationResult(
                overall_score=0.0,
                is_valid=False,
            )

        # Layer 1: Sentence-level grounding check
        sentence_results, grounded_ratio, hallucination_count = self._check_sentence_grounding(
            source_sents, output_sents, nli_model
        )

        # Layer 2: Content word coverage
        content_coverage, missing_words = self._check_content_coverage(source, output, nlp)

        # Layer 3: Named entity verification
        entity_coverage, missing_ents, fabricated_ents = self._check_entity_coverage(
            source, output, nlp
        )

        # Layer 4: Bidirectional entailment (paragraph level)
        forward_ent, backward_ent = self._check_bidirectional_entailment(
            source, output, nli_model
        )

        # Compute overall score with penalties for hallucinations
        base_score = (
            self.grounding_weight * grounded_ratio +
            self.content_coverage_weight * content_coverage +
            self.entity_coverage_weight * entity_coverage
        )

        # Heavy penalty for hallucinations
        hallucination_penalty = min(0.5, hallucination_count * 0.15)

        # Penalty for fabricated entities (strong signal of hallucination)
        entity_penalty = min(0.3, len(fabricated_ents) * 0.1)

        overall_score = max(0.0, base_score - hallucination_penalty - entity_penalty)

        return VerificationResult(
            overall_score=overall_score,
            is_valid=overall_score >= threshold,
            sentence_scores=sentence_results,
            grounded_ratio=grounded_ratio,
            hallucination_count=hallucination_count,
            content_word_coverage=content_coverage,
            missing_content_words=missing_words[:10],  # Limit for readability
            entity_coverage=entity_coverage,
            missing_entities=missing_ents,
            fabricated_entities=fabricated_ents,
            forward_entailment=forward_ent,
            backward_entailment=backward_ent,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        nlp = _get_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _check_sentence_grounding(
        self,
        source_sents: List[str],
        output_sents: List[str],
        nli_model,
    ) -> Tuple[List[SentenceVerification], float, int]:
        """Check if each output sentence is grounded in source using content word overlap.

        NLI models conflate "topically related" with "entailed", so we use
        a stricter content-word-based approach:
        - Extract lemmatized content words from each sentence
        - Compute bidirectional overlap with best-matching source sentence
        - Flag sentences with low overlap as hallucinations

        This catches fabricated content that shares vocabulary but different facts.
        """
        nlp = _get_nlp()

        results = []
        grounded_count = 0
        hallucination_count = 0

        # Pre-compute content words for all source sentences
        source_content_words = []
        for src in source_sents:
            doc = nlp(src)
            words = set()
            for token in doc:
                if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM'} and not token.is_stop:
                    words.add(token.lemma_.lower())
            source_content_words.append(words)

        # Also get all source content words combined for novelty detection
        all_source_words = set()
        for words in source_content_words:
            all_source_words.update(words)

        for out_sent in output_sents:
            # Skip very short sentences
            if len(out_sent.split()) < 4:
                results.append(SentenceVerification(
                    output_sentence=out_sent,
                    best_source_match=None,
                    entailment_score=1.0,
                    is_grounded=True,
                    is_hallucination=False,
                ))
                grounded_count += 1
                continue

            # Get content words for output sentence
            out_doc = nlp(out_sent)
            out_words = set()
            for token in out_doc:
                if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM'} and not token.is_stop:
                    out_words.add(token.lemma_.lower())

            if not out_words:
                results.append(SentenceVerification(
                    output_sentence=out_sent,
                    best_source_match=None,
                    entailment_score=1.0,
                    is_grounded=True,
                    is_hallucination=False,
                ))
                grounded_count += 1
                continue

            # Find best matching source sentence by content word overlap
            best_score = 0.0
            best_match = None
            best_idx = -1

            for i, (src, src_words) in enumerate(zip(source_sents, source_content_words)):
                if not src_words:
                    continue

                # Bidirectional overlap: how much of output is in source AND how much of source is in output
                overlap_words = out_words & src_words

                # Precision: what fraction of output words are from source
                precision = len(overlap_words) / len(out_words) if out_words else 0

                # Recall: what fraction of source words are covered
                recall = len(overlap_words) / len(src_words) if src_words else 0

                # F1-like score
                if precision + recall > 0:
                    score = 2 * precision * recall / (precision + recall)
                else:
                    score = 0

                if score > best_score:
                    best_score = score
                    best_match = src
                    best_idx = i

            # Check for novelty: words in output not anywhere in source
            novel_words = out_words - all_source_words
            novelty_ratio = len(novel_words) / len(out_words) if out_words else 0

            # Grounding criteria:
            # - Best match score >= threshold, OR
            # - Low novelty (most words come from source)
            is_grounded = best_score >= self.grounding_threshold or novelty_ratio < 0.3

            # Hallucination detection (strict):
            # - Very low overlap with ANY source sentence
            # - AND high novelty (many words not in source)
            # - AND substantial sentence
            is_hallucination = (
                best_score < 0.15 and
                novelty_ratio > 0.5 and
                len(out_sent.split()) > 8
            )

            if is_grounded:
                grounded_count += 1
            if is_hallucination:
                hallucination_count += 1
                logger.debug(
                    f"Hallucination detected: '{out_sent[:60]}...' "
                    f"(overlap={best_score:.2f}, novelty={novelty_ratio:.0%})"
                )
                logger.debug(f"  Novel words: {list(novel_words)[:10]}")

            results.append(SentenceVerification(
                output_sentence=out_sent,
                best_source_match=best_match,
                entailment_score=best_score,
                is_grounded=is_grounded,
                is_hallucination=is_hallucination,
            ))

        grounded_ratio = grounded_count / len(output_sents) if output_sents else 0.0
        return results, grounded_ratio, hallucination_count

    def _check_content_coverage(
        self,
        source: str,
        output: str,
        nlp,
    ) -> Tuple[float, List[str]]:
        """Check what fraction of source content words appear in output."""
        source_doc = nlp(source)
        output_doc = nlp(output)

        # Extract content words (nouns, verbs, adjectives, adverbs)
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

        def get_content_words(doc) -> Set[str]:
            words = set()
            for token in doc:
                if token.pos_ in content_pos and not token.is_stop and len(token.text) > 2:
                    words.add(token.lemma_.lower())
            return words

        source_words = get_content_words(source_doc)
        output_words = get_content_words(output_doc)

        if not source_words:
            return 1.0, []

        covered = source_words & output_words
        missing = list(source_words - output_words)
        coverage = len(covered) / len(source_words)

        return coverage, missing

    def _check_entity_coverage(
        self,
        source: str,
        output: str,
        nlp,
    ) -> Tuple[float, List[str], List[str]]:
        """Check named entity coverage and detect fabricated entities.

        Uses fuzzy matching to handle morphological variations:
        - marxist/marxism, german/germanic (shared stems)
        - plural/singular forms
        - lemmatized forms

        Strict only on:
        - Dates and years (common hallucination target)
        - Citations and references (e.g., "(Smith 2023)")
        """
        import re

        source_doc = nlp(source)
        output_doc = nlp(output)

        # Extract named entities with lemmas for fuzzy matching
        def get_entities_with_stems(doc) -> Tuple[Set[str], Set[str]]:
            """Return (entities, stems) for fuzzy matching."""
            entities = set()
            stems = set()
            for ent in doc.ents:
                ent_text = ent.text.strip().lower()
                if len(ent_text) > 1:
                    entities.add(ent_text)
                    # Add stem (first 4+ chars) for fuzzy matching
                    # This handles marxist/marxism, german/germanic
                    stem = self._get_entity_stem(ent_text)
                    if stem:
                        stems.add(stem)
            return entities, stems

        source_ents, source_stems = get_entities_with_stems(source_doc)
        output_ents, output_stems = get_entities_with_stems(output_doc)

        # Also extract years (4-digit numbers in parentheses or standalone)
        # These are common hallucination targets for academic text
        year_pattern = re.compile(r'\b(1[89]\d{2}|20[0-2]\d)\b')  # 1800-2029
        source_years = set(year_pattern.findall(source))
        output_years = set(year_pattern.findall(output))

        # Extract citation-like patterns: (Author Year), (cf. X), etc.
        citation_pattern = re.compile(r'\([^)]*(?:19|20)\d{2}[^)]*\)')
        source_citations = set(citation_pattern.findall(source))
        output_citations = set(citation_pattern.findall(output))

        # Fabricated years are a strong hallucination signal
        fabricated_years = output_years - source_years
        fabricated_citations = output_citations - source_citations

        if fabricated_years:
            logger.warning(f"Fabricated years detected: {fabricated_years}")
        if fabricated_citations:
            logger.warning(f"Fabricated citations detected: {fabricated_citations}")

        if not source_ents:
            # No entities in source - check if output has fabricated ones
            # But filter out entities that share stems with source text words
            source_text_stems = self._get_text_stems(source, nlp)
            truly_fabricated = [
                e for e in output_ents
                if not self._entity_matches_any_stem(e, source_text_stems)
            ]
            truly_fabricated.extend([f"year:{y}" for y in fabricated_years])
            return 1.0, [], truly_fabricated

        # Use fuzzy matching: entity is covered if exact match OR stem match
        covered = set()
        for src_ent in source_ents:
            if src_ent in output_ents:
                covered.add(src_ent)
            elif self._entity_matches_any_stem(src_ent, output_stems):
                covered.add(src_ent)

        missing = list(source_ents - covered)

        # Fabricated = output entities not matching any source entity or stem
        fabricated = []
        for out_ent in output_ents:
            if out_ent not in source_ents:
                if not self._entity_matches_any_stem(out_ent, source_stems):
                    fabricated.append(out_ent)

        # Add fabricated years/citations to fabricated list (strong penalty)
        fabricated.extend([f"year:{y}" for y in fabricated_years])
        fabricated.extend([f"citation:{c[:30]}" for c in fabricated_citations])

        coverage = len(covered) / len(source_ents) if source_ents else 1.0

        return coverage, missing, fabricated

    def _get_entity_stem(self, entity: str) -> Optional[str]:
        """Get stem of entity for fuzzy matching.

        Returns first 4+ characters, handling common suffixes.
        """
        if len(entity) < 4:
            return None

        # Common suffixes to strip for stem matching
        suffixes = ['ism', 'ist', 'ists', 'ic', 'ics', 'ian', 'ians', 'ese', 'ish', 's']

        stem = entity.lower()
        for suffix in sorted(suffixes, key=len, reverse=True):
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
                stem = stem[:-len(suffix)]
                break

        return stem if len(stem) >= 4 else entity[:4]

    def _entity_matches_any_stem(self, entity: str, stems: Set[str]) -> bool:
        """Check if entity matches any stem in the set."""
        entity_stem = self._get_entity_stem(entity)
        if not entity_stem:
            return False

        # Direct stem match
        if entity_stem in stems:
            return True

        # Check if any stem is a prefix of entity or vice versa
        for stem in stems:
            if stem.startswith(entity_stem) or entity_stem.startswith(stem):
                return True

        return False

    def _get_text_stems(self, text: str, nlp) -> Set[str]:
        """Extract stems from all content words in text."""
        doc = nlp(text)
        stems = set()
        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN', 'ADJ'} and len(token.text) >= 4:
                stem = self._get_entity_stem(token.text.lower())
                if stem:
                    stems.add(stem)
        return stems

    def _check_bidirectional_entailment(
        self,
        source: str,
        output: str,
        nli_model,
    ) -> Tuple[float, float]:
        """Check paragraph-level bidirectional entailment."""
        import numpy as np

        if not nli_model:
            return 0.5, 0.5

        def get_entailment_score(premise: str, hypothesis: str) -> float:
            scores = nli_model.predict([(premise, hypothesis)], show_progress_bar=False)
            if len(scores.shape) > 1:
                logits = scores[0]
            else:
                logits = scores
            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / np.sum(exp_scores)
            return float(probs[2]) if len(probs) > 2 else float(probs[-1])

        # Truncate for model limits
        source_truncated = source[:1500]
        output_truncated = output[:1500]

        forward = get_entailment_score(source_truncated, output_truncated)
        backward = get_entailment_score(output_truncated, source_truncated)

        return forward, backward


# Singleton instance
_verifier = None


def get_semantic_verifier() -> SemanticVerifier:
    """Get or create the singleton semantic verifier."""
    global _verifier
    if _verifier is None:
        _verifier = SemanticVerifier()
    return _verifier


def verify_semantic_preservation(
    source: str,
    output: str,
    threshold: float = 0.7,
) -> VerificationResult:
    """Convenience function for semantic verification.

    Args:
        source: Original source text.
        output: Generated output text.
        threshold: Score threshold for validity.

    Returns:
        VerificationResult with detailed analysis.
    """
    verifier = get_semantic_verifier()
    return verifier.verify(source, output, threshold)
