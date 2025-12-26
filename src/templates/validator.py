"""Validation and repair for template-based style transfer.

Validates filled sentences and paragraphs against:
- Statistical properties (length, complexity, burstiness)
- Technical term preservation
- Vocabulary distribution
- Grammatical correctness
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import Counter

from ..utils.nlp import get_nlp, split_into_sentences, calculate_burstiness
from ..utils.logging import get_logger
from .models import (
    CorpusStatistics,
    VocabularyProfile,
    ValidationIssue,
    RepairAction,
    SentenceValidationResult,
    ParagraphRepairAction,
    ParagraphValidationResult,
)
from .vocabulary import WordClassifier, GeneralWordMapper
from .statistics import FUNCTION_WORDS

logger = get_logger(__name__)


class SentenceValidator:
    """Validates individual sentences against style requirements."""

    def __init__(
        self,
        corpus_stats: CorpusStatistics,
        vocabulary: Optional[VocabularyProfile] = None
    ):
        """Initialize validator.

        Args:
            corpus_stats: Target author's corpus statistics.
            vocabulary: Target author's vocabulary profile.
        """
        self._nlp = None
        self.stats = corpus_stats
        self.vocabulary = vocabulary
        self.classifier = WordClassifier()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def validate(
        self,
        sentence: str,
        required_terms: Optional[Set[str]] = None,
        position_in_paragraph: int = 0,
        total_sentences: int = 1
    ) -> SentenceValidationResult:
        """Validate a sentence.

        Args:
            sentence: The sentence to validate.
            required_terms: Technical terms that must appear.
            position_in_paragraph: Position (0-indexed).
            total_sentences: Total sentences in paragraph.

        Returns:
            SentenceValidationResult with issues and repair suggestions.
        """
        logger.debug(f"Validating sentence: '{sentence[:50]}...'")

        required_terms = required_terms or set()
        issues = []
        repairs = []

        # Parse sentence
        doc = self.nlp(sentence)
        word_count = len([t for t in doc if not t.is_punct and not t.is_space])
        logger.debug(f"  Word count: {word_count}, Required terms: {len(required_terms)}")

        # Check length
        length_result = self._validate_length(word_count)
        if not length_result[0]:
            issues.append(length_result[1])
            repairs.append(length_result[2])

        # Check complexity
        complexity = self._compute_complexity(doc)
        complexity_result = self._validate_complexity(complexity)
        if not complexity_result[0]:
            issues.append(complexity_result[1])
            repairs.append(complexity_result[2])

        # Check technical term preservation
        terms_result = self._validate_technical_terms(sentence, required_terms)
        if not terms_result[0]:
            issues.append(terms_result[1])
            repairs.append(terms_result[2])

        # Check vocabulary
        vocab_result = self._validate_vocabulary(sentence)
        if not vocab_result[0]:
            issues.append(vocab_result[1])
            repairs.append(vocab_result[2])

        # Check grammar
        grammar_result = self._validate_grammar(doc)
        if not grammar_result[0]:
            issues.append(grammar_result[1])
            repairs.append(grammar_result[2])

        is_valid = len([i for i in issues if i.severity == "HIGH"]) == 0

        logger.debug(f"  Validation result: valid={is_valid}, issues={[i.type for i in issues]}")

        return SentenceValidationResult(
            is_valid=is_valid,
            issues=issues,
            length_ok=length_result[0],
            complexity_ok=complexity_result[0],
            vocabulary_ok=vocab_result[0],
            grammar_ok=grammar_result[0],
            technical_terms_preserved=terms_result[0],
            suggested_repairs=[r for r in repairs if r is not None],
        )

    def _validate_length(
        self,
        word_count: int
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[RepairAction]]:
        """Validate sentence length."""
        mean = self.stats.sentence_length_mean
        std = self.stats.sentence_length_std

        # Allow 2 standard deviations
        min_length = max(3, mean - 2 * std)
        max_length = mean + 2 * std

        if word_count < min_length:
            return (
                False,
                ValidationIssue(
                    type="LENGTH",
                    message=f"Sentence too short ({word_count} words, min {min_length:.0f})",
                    severity="MEDIUM",
                    details={"actual": word_count, "min": min_length, "max": max_length},
                ),
                RepairAction(
                    type="EXTEND_SENTENCE",
                    target_range=(int(min_length), int(max_length)),
                    issues=["too_short"],
                ),
            )

        if word_count > max_length:
            return (
                False,
                ValidationIssue(
                    type="LENGTH",
                    message=f"Sentence too long ({word_count} words, max {max_length:.0f})",
                    severity="MEDIUM",
                    details={"actual": word_count, "min": min_length, "max": max_length},
                ),
                RepairAction(
                    type="SHORTEN_SENTENCE",
                    target_range=(int(min_length), int(max_length)),
                    issues=["too_long"],
                ),
            )

        return (True, None, None)

    def _validate_complexity(
        self,
        complexity: float
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[RepairAction]]:
        """Validate sentence complexity."""
        mean = self.stats.complexity_mean
        std = self.stats.complexity_std

        min_complexity = max(1, mean - 2 * std)
        max_complexity = mean + 2 * std

        if complexity < min_complexity:
            return (
                False,
                ValidationIssue(
                    type="COMPLEXITY",
                    message=f"Sentence too simple (complexity {complexity:.1f}, min {min_complexity:.1f})",
                    severity="LOW",
                    details={"actual": complexity, "min": min_complexity, "max": max_complexity},
                ),
                RepairAction(
                    type="INCREASE_COMPLEXITY",
                    issues=["too_simple"],
                ),
            )

        if complexity > max_complexity:
            return (
                False,
                ValidationIssue(
                    type="COMPLEXITY",
                    message=f"Sentence too complex (complexity {complexity:.1f}, max {max_complexity:.1f})",
                    severity="LOW",
                    details={"actual": complexity, "min": min_complexity, "max": max_complexity},
                ),
                RepairAction(
                    type="DECREASE_COMPLEXITY",
                    issues=["too_complex"],
                ),
            )

        return (True, None, None)

    def _validate_technical_terms(
        self,
        sentence: str,
        required_terms: Set[str]
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[RepairAction]]:
        """Validate that required technical terms are present."""
        if not required_terms:
            return (True, None, None)

        sentence_lower = sentence.lower()
        missing = []

        for term in required_terms:
            if term.lower() not in sentence_lower:
                missing.append(term)

        if missing:
            return (
                False,
                ValidationIssue(
                    type="TECHNICAL_TERMS",
                    message=f"Missing technical terms: {', '.join(missing)}",
                    severity="HIGH",
                    details={"missing": missing, "required": list(required_terms)},
                ),
                RepairAction(
                    type="INSERT_TERMS",
                    terms=missing,
                    issues=["missing_terms"],
                ),
            )

        return (True, None, None)

    def _validate_vocabulary(
        self,
        sentence: str
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[RepairAction]]:
        """Validate vocabulary usage."""
        if not self.vocabulary:
            return (True, None, None)

        doc = self.nlp(sentence)
        non_author_words = []

        for token in doc:
            if token.is_punct or token.is_space or token.is_stop:
                continue

            word = token.text.lower()
            lemma = token.lemma_.lower()

            # Check if word is in author's vocabulary
            in_vocab = (
                word in self.vocabulary.general_words or
                lemma in self.vocabulary.general_words or
                word in self.vocabulary.common_verbs or
                lemma in self.vocabulary.common_verbs or
                word in self.vocabulary.modifiers or
                lemma in self.vocabulary.modifiers or
                word in FUNCTION_WORDS
            )

            # Check if it's technical (allowed)
            classification = self.classifier.classify(token.text, sentence)
            is_technical = classification.word_type.value == "technical"

            if not in_vocab and not is_technical:
                non_author_words.append(word)

        # Allow up to 30% non-author words
        word_count = len([t for t in doc if not t.is_punct and not t.is_space])
        if word_count > 0:
            non_author_ratio = len(non_author_words) / word_count

            if non_author_ratio > 0.3:
                return (
                    False,
                    ValidationIssue(
                        type="VOCABULARY",
                        message=f"Too many non-author words ({len(non_author_words)}/{word_count})",
                        severity="MEDIUM",
                        details={"non_author_words": non_author_words[:10], "ratio": non_author_ratio},
                    ),
                    RepairAction(
                        type="SUBSTITUTE_WORDS",
                        words=non_author_words,
                        issues=["vocabulary_mismatch"],
                    ),
                )

        return (True, None, None)

    def _validate_grammar(
        self,
        doc
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[RepairAction]]:
        """Validate basic grammar."""
        issues_found = []

        # Check for ROOT verb
        has_root = any(t.dep_ == "ROOT" for t in doc)
        if not has_root:
            issues_found.append("no_main_verb")

        # Check for subject
        has_subject = any(t.dep_ in ("nsubj", "nsubjpass", "expl") for t in doc)
        if not has_subject and len(doc) > 3:
            issues_found.append("no_subject")

        # Check for unbalanced parentheses/quotes
        text = doc.text
        if text.count('(') != text.count(')'):
            issues_found.append("unbalanced_parentheses")
        if text.count('"') % 2 != 0:
            issues_found.append("unbalanced_quotes")

        if issues_found:
            return (
                False,
                ValidationIssue(
                    type="GRAMMAR",
                    message=f"Grammar issues: {', '.join(issues_found)}",
                    severity="HIGH" if "no_main_verb" in issues_found else "MEDIUM",
                    details={"issues": issues_found},
                ),
                RepairAction(
                    type="FIX_GRAMMAR",
                    issues=issues_found,
                ),
            )

        return (True, None, None)

    def _compute_complexity(self, doc) -> float:
        """Compute syntactic complexity."""
        max_depth = 0
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 20:
                    break
            max_depth = max(max_depth, depth)
        return float(max_depth)


class SentenceRepairer:
    """Repairs sentences based on validation issues."""

    def __init__(
        self,
        corpus_stats: CorpusStatistics,
        vocabulary: Optional[VocabularyProfile] = None
    ):
        """Initialize repairer.

        Args:
            corpus_stats: Target author's statistics.
            vocabulary: Target author's vocabulary.
        """
        self._nlp = None
        self.stats = corpus_stats
        self.vocabulary = vocabulary
        self.mapper = GeneralWordMapper(vocabulary) if vocabulary else None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def repair(
        self,
        sentence: str,
        validation_result: SentenceValidationResult,
        required_terms: Optional[Set[str]] = None,
        max_iterations: int = 3
    ) -> Tuple[str, bool]:
        """Repair a sentence based on validation issues.

        Args:
            sentence: Sentence to repair.
            validation_result: Validation result with issues.
            required_terms: Terms that must be preserved.
            max_iterations: Maximum repair iterations.

        Returns:
            Tuple of (repaired_sentence, success).
        """
        required_terms = required_terms or set()
        current = sentence
        iteration = 0

        while iteration < max_iterations:
            made_changes = False

            for repair in validation_result.suggested_repairs:
                new_sentence = self._apply_repair(current, repair, required_terms)
                if new_sentence != current:
                    current = new_sentence
                    made_changes = True

            if not made_changes:
                break

            iteration += 1

        # Final validation
        validator = SentenceValidator(self.stats, self.vocabulary)
        final_result = validator.validate(current, required_terms)

        return current, final_result.is_valid

    def _apply_repair(
        self,
        sentence: str,
        repair: RepairAction,
        required_terms: Set[str]
    ) -> str:
        """Apply a single repair action.

        Args:
            sentence: Current sentence.
            repair: Repair action to apply.
            required_terms: Terms to preserve.

        Returns:
            Repaired sentence.
        """
        if repair.type == "INSERT_TERMS":
            return self._insert_terms(sentence, repair.terms, required_terms)

        elif repair.type == "SUBSTITUTE_WORDS":
            return self._substitute_words(sentence, repair.words, required_terms)

        elif repair.type == "EXTEND_SENTENCE":
            return self._extend_sentence(sentence, repair.target_range)

        elif repair.type == "SHORTEN_SENTENCE":
            return self._shorten_sentence(sentence, repair.target_range)

        elif repair.type == "FIX_GRAMMAR":
            return self._fix_grammar(sentence, repair.issues)

        return sentence

    def _insert_terms(
        self,
        sentence: str,
        terms: List[str],
        required_terms: Set[str]
    ) -> str:
        """Insert missing terms into sentence."""
        doc = self.nlp(sentence)

        for term in terms:
            if term.lower() not in sentence.lower():
                # Find a good insertion point
                # Try to insert near related words or at the end
                insertion_point = self._find_insertion_point(doc, term)

                if insertion_point > 0:
                    words = sentence.split()
                    words.insert(insertion_point, term)
                    sentence = " ".join(words)
                else:
                    # Insert before final punctuation
                    if sentence and sentence[-1] in ".!?":
                        sentence = sentence[:-1] + f" including {term}" + sentence[-1]
                    else:
                        sentence += f" regarding {term}"

        return sentence

    def _find_insertion_point(self, doc, term: str) -> int:
        """Find best insertion point for a term."""
        # Look for prepositions or noun phrases where term fits
        for i, token in enumerate(doc):
            if token.dep_ in ("prep", "pobj") and i < len(doc) - 1:
                return i + 1

        # Default: insert before last noun phrase
        for i, token in reversed(list(enumerate(doc))):
            if token.pos_ == "NOUN":
                return i

        return -1

    def _substitute_words(
        self,
        sentence: str,
        words_to_replace: List[str],
        required_terms: Set[str]
    ) -> str:
        """Substitute non-author words with author vocabulary."""
        if not self.mapper:
            return sentence

        doc = self.nlp(sentence)
        result_tokens = []

        for token in doc:
            word = token.text
            word_lower = word.lower()

            # Don't replace required terms
            if word_lower in {t.lower() for t in required_terms}:
                result_tokens.append(word)
                continue

            # Check if this word needs replacement
            if word_lower in words_to_replace:
                mapping = self.mapper.map_word(word, token.pos_)
                if mapping and mapping.similarity > 0.3:
                    new_word = mapping.target_word
                    # Preserve case
                    if word[0].isupper():
                        new_word = new_word.capitalize()
                    result_tokens.append(new_word)
                    continue

            result_tokens.append(word)

        return " ".join(result_tokens)

    def _extend_sentence(
        self,
        sentence: str,
        target_range: Tuple[int, int]
    ) -> str:
        """Extend a too-short sentence."""
        doc = self.nlp(sentence)
        word_count = len([t for t in doc if not t.is_punct and not t.is_space])

        if word_count >= target_range[0]:
            return sentence

        # Add modifiers or prepositional phrases
        words = []
        for token in doc:
            words.append(token.text)

            # Add modifier before nouns
            if (token.pos_ == "NOUN" and
                not any(c.pos_ == "ADJ" for c in token.children) and
                self.vocabulary):

                # Get a modifier from vocabulary
                modifiers = list(self.vocabulary.modifiers.keys())
                if modifiers:
                    modifier = modifiers[0]  # Simple selection
                    words.insert(-1, modifier)
                    word_count += 1

                    if word_count >= target_range[0]:
                        break

        return " ".join(words)

    def _shorten_sentence(
        self,
        sentence: str,
        target_range: Tuple[int, int]
    ) -> str:
        """Shorten a too-long sentence."""
        doc = self.nlp(sentence)
        word_count = len([t for t in doc if not t.is_punct and not t.is_space])

        if word_count <= target_range[1]:
            return sentence

        # Remove non-essential modifiers
        essential_deps = {"ROOT", "nsubj", "nsubjpass", "dobj", "pobj"}
        words = []

        for token in doc:
            # Keep essential words
            if token.dep_ in essential_deps or token.is_punct:
                words.append(token.text)
            # Keep some modifiers
            elif token.pos_ == "ADJ" and len(words) > 0:
                # Keep first adjective, skip others
                words.append(token.text)
            elif token.pos_ not in ("ADJ", "ADV"):
                words.append(token.text)

        result = " ".join(words)

        # Clean up
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,;:!?])', r'\1', result)

        return result.strip()

    def _fix_grammar(
        self,
        sentence: str,
        issues: List[str]
    ) -> str:
        """Fix grammar issues."""
        result = sentence

        if "unbalanced_parentheses" in issues:
            # Remove all parentheses
            result = re.sub(r'[()]', '', result)

        if "unbalanced_quotes" in issues:
            # Remove all quotes
            result = re.sub(r'["\']', '', result)

        # Clean up
        result = re.sub(r'\s+', ' ', result)
        return result.strip()


class ParagraphValidator:
    """Validates entire paragraphs against style requirements."""

    def __init__(
        self,
        corpus_stats: CorpusStatistics,
        vocabulary: Optional[VocabularyProfile] = None
    ):
        """Initialize validator.

        Args:
            corpus_stats: Target author's statistics.
            vocabulary: Target author's vocabulary.
        """
        self._nlp = None
        self.stats = corpus_stats
        self.vocabulary = vocabulary
        self.sentence_validator = SentenceValidator(corpus_stats, vocabulary)

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def validate(
        self,
        paragraph: str,
        required_terms: Optional[Set[str]] = None
    ) -> ParagraphValidationResult:
        """Validate a paragraph.

        Args:
            paragraph: The paragraph to validate.
            required_terms: Technical terms that must appear.

        Returns:
            ParagraphValidationResult with issues and repairs.
        """
        required_terms = required_terms or set()
        issues = []
        repairs = []

        sentences = split_into_sentences(paragraph)
        if not sentences:
            return ParagraphValidationResult(is_valid=False, issues=[
                ValidationIssue(
                    type="STRUCTURE",
                    message="No sentences found in paragraph",
                    severity="HIGH",
                )
            ])

        # Check burstiness
        burstiness_result = self._validate_burstiness(sentences)
        if not burstiness_result[0]:
            issues.append(burstiness_result[1])
            repairs.append(burstiness_result[2])

        # Check rhythm pattern
        rhythm_result = self._validate_rhythm(sentences)
        if not rhythm_result[0]:
            issues.append(rhythm_result[1])
            repairs.append(rhythm_result[2])

        # Check vocabulary distribution
        vocab_result = self._validate_vocabulary_distribution(paragraph)
        if not vocab_result[0]:
            issues.append(vocab_result[1])
            repairs.append(vocab_result[2])

        # Check technical term coverage
        terms_result = self._validate_term_coverage(paragraph, required_terms)
        if not terms_result[0]:
            issues.append(terms_result[1])
            repairs.append(terms_result[2])

        # Compute corpus similarity
        corpus_similarity = self._compute_corpus_similarity(sentences)

        is_valid = len([i for i in issues if i.severity == "HIGH"]) == 0

        return ParagraphValidationResult(
            is_valid=is_valid,
            issues=issues,
            burstiness_ok=burstiness_result[0],
            rhythm_pattern_ok=rhythm_result[0],
            vocabulary_distribution_ok=vocab_result[0],
            structure_ok=True,  # Basic structure check
            corpus_similarity=corpus_similarity,
            suggested_repairs=[r for r in repairs if r is not None],
        )

    def _validate_burstiness(
        self,
        sentences: List[str]
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[ParagraphRepairAction]]:
        """Validate sentence length variation (burstiness)."""
        if len(sentences) < 2:
            return (True, None, None)

        burstiness = calculate_burstiness(sentences)
        target_mean = self.stats.burstiness_mean
        target_std = self.stats.burstiness_std

        min_burstiness = max(0, target_mean - 2 * target_std)
        max_burstiness = target_mean + 2 * target_std

        if burstiness < min_burstiness or burstiness > max_burstiness:
            lengths = [len(s.split()) for s in sentences]
            return (
                False,
                ValidationIssue(
                    type="BURSTINESS",
                    message=f"Burstiness {burstiness:.2f} outside range [{min_burstiness:.2f}, {max_burstiness:.2f}]",
                    severity="MEDIUM",
                    details={
                        "actual": burstiness,
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "sentence_lengths": lengths,
                    },
                ),
                ParagraphRepairAction(
                    type="ADJUST_BURSTINESS",
                    current=burstiness,
                    target=target_mean,
                    lengths=lengths,
                ),
            )

        return (True, None, None)

    def _validate_rhythm(
        self,
        sentences: List[str]
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[ParagraphRepairAction]]:
        """Validate rhythm pattern matches author's patterns."""
        if len(sentences) < 3 or not self.stats.rhythm_patterns:
            return (True, None, None)

        lengths = [len(s.split()) for s in sentences]
        mean_length = sum(lengths) / len(lengths)

        if mean_length == 0:
            return (True, None, None)

        current_pattern = [l / mean_length for l in lengths]

        # Find best matching pattern
        best_match_score = 0.0
        best_pattern = None

        for pattern in self.stats.rhythm_patterns:
            if len(pattern) == len(current_pattern):
                # Compute similarity
                diff = sum(abs(a - b) for a, b in zip(current_pattern, pattern))
                similarity = 1 / (1 + diff)

                if similarity > best_match_score:
                    best_match_score = similarity
                    best_pattern = pattern

        if best_match_score < 0.5:
            return (
                False,
                ValidationIssue(
                    type="RHYTHM",
                    message=f"Rhythm pattern doesn't match author's style (similarity {best_match_score:.2f})",
                    severity="LOW",
                    details={
                        "current_pattern": [round(p, 2) for p in current_pattern],
                        "best_match": best_pattern,
                        "match_score": best_match_score,
                    },
                ),
                ParagraphRepairAction(
                    type="ADJUST_RHYTHM",
                    current_pattern=current_pattern,
                    target_patterns=self.stats.rhythm_patterns[:3],
                ),
            )

        return (True, None, None)

    def _validate_vocabulary_distribution(
        self,
        paragraph: str
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[ParagraphRepairAction]]:
        """Validate vocabulary matches author's distribution."""
        if not self.vocabulary or not self.stats.general_word_frequencies:
            return (True, None, None)

        doc = self.nlp(paragraph)

        # Count word usage
        word_counts = Counter()
        total_words = 0

        for token in doc:
            if token.is_punct or token.is_space or token.is_stop:
                continue
            lemma = token.lemma_.lower()
            word_counts[lemma] += 1
            total_words += 1

        if total_words == 0:
            return (True, None, None)

        # Compare to author distribution
        overused = []
        underused = []

        for word, count in word_counts.items():
            actual_freq = count / total_words
            target_freq = self.stats.general_word_frequencies.get(word, 0)

            if target_freq > 0:
                ratio = actual_freq / target_freq
                if ratio > 2.0:
                    overused.append((word, actual_freq, target_freq))
                elif ratio < 0.5 and target_freq > 0.01:
                    underused.append((word, actual_freq, target_freq))

        if overused or underused:
            return (
                False,
                ValidationIssue(
                    type="VOCABULARY_DISTRIBUTION",
                    message=f"Vocabulary distribution differs from author ({len(overused)} overused, {len(underused)} underused)",
                    severity="LOW",
                    details={"overused": overused[:5], "underused": underused[:5]},
                ),
                ParagraphRepairAction(
                    type="ADJUST_VOCABULARY",
                    overused=overused[:10],
                    underused=underused[:10],
                ),
            )

        return (True, None, None)

    def _validate_term_coverage(
        self,
        paragraph: str,
        required_terms: Set[str]
    ) -> Tuple[bool, Optional[ValidationIssue], Optional[ParagraphRepairAction]]:
        """Validate all required terms appear."""
        if not required_terms:
            return (True, None, None)

        paragraph_lower = paragraph.lower()
        missing = [t for t in required_terms if t.lower() not in paragraph_lower]

        if missing:
            return (
                False,
                ValidationIssue(
                    type="TERM_COVERAGE",
                    message=f"Missing required terms: {', '.join(missing)}",
                    severity="HIGH",
                    details={"missing": missing},
                ),
                None,  # Will be handled by sentence-level repairs
            )

        return (True, None, None)

    def _compute_corpus_similarity(self, sentences: List[str]) -> float:
        """Compute similarity to corpus feature distribution."""
        if self.stats.paragraph_feature_mean is None:
            return 0.5  # Default

        import numpy as np

        # Compute features
        lengths = [len(s.split()) for s in sentences]
        mean_length = np.mean(lengths) if lengths else 0
        std_length = np.std(lengths) if len(lengths) > 1 else 0
        burstiness = calculate_burstiness(sentences) if len(sentences) > 1 else 0

        # Simplified feature vector (matches _compute_paragraph_features in statistics.py)
        features = np.array([
            len(sentences),
            mean_length,
            std_length,
            burstiness,
            3.0,  # Placeholder for complexity
            lengths[0] / mean_length if mean_length > 0 else 1.0,
            lengths[-1] / mean_length if mean_length > 0 else 1.0,
        ])

        # Compute distance from mean
        diff = features - self.stats.paragraph_feature_mean
        if self.stats.paragraph_feature_std is not None:
            # Normalize by std
            normalized_diff = diff / (self.stats.paragraph_feature_std + 1e-6)
            distance = np.linalg.norm(normalized_diff)
        else:
            distance = np.linalg.norm(diff)

        # Convert to similarity (0-1)
        similarity = 1 / (1 + distance * 0.1)
        return float(np.clip(similarity, 0, 1))


class ParagraphRepairer:
    """Repairs paragraphs based on validation issues."""

    def __init__(
        self,
        corpus_stats: CorpusStatistics,
        vocabulary: Optional[VocabularyProfile] = None
    ):
        """Initialize repairer.

        Args:
            corpus_stats: Target author's statistics.
            vocabulary: Target author's vocabulary.
        """
        self._nlp = None
        self.stats = corpus_stats
        self.vocabulary = vocabulary
        self.sentence_repairer = SentenceRepairer(corpus_stats, vocabulary)

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def repair(
        self,
        paragraph: str,
        validation_result: ParagraphValidationResult,
        required_terms: Optional[Set[str]] = None,
        max_iterations: int = 3
    ) -> Tuple[str, bool]:
        """Repair a paragraph based on validation issues.

        Args:
            paragraph: Paragraph to repair.
            validation_result: Validation result with issues.
            required_terms: Terms that must be preserved.
            max_iterations: Maximum iterations.

        Returns:
            Tuple of (repaired_paragraph, success).
        """
        required_terms = required_terms or set()
        current = paragraph
        iteration = 0

        while iteration < max_iterations:
            made_changes = False

            for repair in validation_result.suggested_repairs:
                new_paragraph = self._apply_repair(current, repair, required_terms)
                if new_paragraph != current:
                    current = new_paragraph
                    made_changes = True

            if not made_changes:
                break

            iteration += 1

        # Final validation
        validator = ParagraphValidator(self.stats, self.vocabulary)
        final_result = validator.validate(current, required_terms)

        return current, final_result.is_valid

    def _apply_repair(
        self,
        paragraph: str,
        repair: ParagraphRepairAction,
        required_terms: Set[str]
    ) -> str:
        """Apply a paragraph-level repair."""
        if repair.type == "ADJUST_BURSTINESS":
            return self._adjust_burstiness(paragraph, repair)

        elif repair.type == "ADJUST_RHYTHM":
            return self._adjust_rhythm(paragraph, repair)

        elif repair.type == "ADJUST_VOCABULARY":
            return self._adjust_vocabulary(paragraph, repair, required_terms)

        return paragraph

    def _adjust_burstiness(
        self,
        paragraph: str,
        repair: ParagraphRepairAction
    ) -> str:
        """Adjust sentence length variation."""
        sentences = split_into_sentences(paragraph)
        if len(sentences) < 2:
            return paragraph

        current = repair.current
        target = repair.target

        if current < target:
            # Need more variation - extend or shorten some sentences
            # Find sentences close to mean and modify them
            lengths = [len(s.split()) for s in sentences]
            mean_length = sum(lengths) / len(lengths)

            new_sentences = []
            for i, (sent, length) in enumerate(zip(sentences, lengths)):
                ratio = length / mean_length if mean_length > 0 else 1

                if 0.8 < ratio < 1.2:  # Close to mean
                    if i % 2 == 0:
                        # Make shorter by removing modifiers
                        sent = self._remove_modifiers(sent)
                    else:
                        # Make longer by adding modifiers
                        sent = self._add_modifiers(sent)

                new_sentences.append(sent)

            return " ".join(new_sentences)

        return paragraph

    def _adjust_rhythm(
        self,
        paragraph: str,
        repair: ParagraphRepairAction
    ) -> str:
        """Adjust rhythm to match author patterns."""
        sentences = split_into_sentences(paragraph)
        if not sentences or not repair.target_patterns:
            return paragraph

        # Find best matching target pattern
        target = None
        for pattern in repair.target_patterns:
            if len(pattern) == len(sentences):
                target = pattern
                break

        if not target:
            return paragraph

        # Adjust each sentence length to match pattern
        lengths = [len(s.split()) for s in sentences]
        mean_length = sum(lengths) / len(lengths)

        new_sentences = []
        for sent, current_ratio, target_ratio in zip(sentences, repair.current_pattern, target):
            target_length = int(target_ratio * mean_length)
            current_length = len(sent.split())

            if current_length < target_length:
                sent = self._extend_to_length(sent, target_length)
            elif current_length > target_length:
                sent = self._shorten_to_length(sent, target_length)

            new_sentences.append(sent)

        return " ".join(new_sentences)

    def _adjust_vocabulary(
        self,
        paragraph: str,
        repair: ParagraphRepairAction,
        required_terms: Set[str]
    ) -> str:
        """Adjust vocabulary to match author's distribution."""
        if not self.vocabulary:
            return paragraph

        # Replace overused words with alternatives
        result = paragraph

        for word, actual_freq, target_freq in repair.overused:
            if word.lower() in {t.lower() for t in required_terms}:
                continue

            # Find alternative
            alternatives = self._find_alternatives(word)
            if alternatives:
                # Replace some occurrences
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                matches = list(pattern.finditer(result))

                if len(matches) > 1:
                    # Replace half the occurrences
                    for match in matches[::2]:
                        result = result[:match.start()] + alternatives[0] + result[match.end():]

        return result

    def _find_alternatives(self, word: str) -> List[str]:
        """Find alternative words from author's vocabulary."""
        if not self.vocabulary:
            return []

        # Look for similar words in vocabulary
        from ..utils.nlp import compute_semantic_similarity

        candidates = []
        all_words = (
            list(self.vocabulary.general_words.keys()) +
            list(self.vocabulary.common_verbs.keys()) +
            list(self.vocabulary.modifiers.keys())
        )

        for candidate in all_words:
            if candidate != word:
                similarity = compute_semantic_similarity(word, candidate)
                if similarity > 0.4:
                    candidates.append((candidate, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:3]]

    def _remove_modifiers(self, sentence: str) -> str:
        """Remove some modifiers from a sentence."""
        doc = self.nlp(sentence)
        tokens = []
        skip_next = False

        for token in doc:
            if skip_next:
                skip_next = False
                continue

            # Remove every other adjective
            if token.pos_ == "ADJ" and len(tokens) > 0:
                continue

            tokens.append(token.text)

        return " ".join(tokens)

    def _add_modifiers(self, sentence: str) -> str:
        """Add modifiers to a sentence."""
        if not self.vocabulary:
            return sentence

        doc = self.nlp(sentence)
        tokens = []

        modifiers = list(self.vocabulary.modifiers.keys())
        if not modifiers:
            return sentence

        mod_index = 0
        for token in doc:
            # Add modifier before nouns that don't have one
            if (token.pos_ == "NOUN" and
                not any(c.pos_ == "ADJ" for c in token.children)):
                tokens.append(modifiers[mod_index % len(modifiers)])
                mod_index += 1

            tokens.append(token.text)

        return " ".join(tokens)

    def _extend_to_length(self, sentence: str, target: int) -> str:
        """Extend sentence to target word count."""
        current = len(sentence.split())
        while current < target:
            sentence = self._add_modifiers(sentence)
            new_len = len(sentence.split())
            if new_len == current:
                break
            current = new_len
        return sentence

    def _shorten_to_length(self, sentence: str, target: int) -> str:
        """Shorten sentence to target word count."""
        current = len(sentence.split())
        while current > target:
            sentence = self._remove_modifiers(sentence)
            new_len = len(sentence.split())
            if new_len == current:
                break
            current = new_len
        return sentence
