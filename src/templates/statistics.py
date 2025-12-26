"""Corpus statistics extraction for template-based style transfer."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..utils.nlp import get_nlp, split_into_sentences, calculate_burstiness
from ..utils.logging import get_logger
from .models import (
    CorpusStatistics,
    VocabularyProfile,
    RhetoricalRole,
    LogicalRelation,
)

logger = get_logger(__name__)


# Common English function words (top ~100)
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "because", "as",
    "of", "to", "in", "for", "on", "with", "at", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "this",
    "that", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "him", "her", "his", "hers", "we", "us", "our", "you",
    "your", "i", "me", "my", "who", "which", "what", "where", "when",
    "how", "why", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "still", "yet",
}

# Connector words by category
CONNECTOR_PATTERNS = {
    "causal": [
        "therefore", "thus", "hence", "consequently", "accordingly",
        "so", "because", "since", "as a result", "for this reason",
        "due to", "owing to", "leads to", "causes", "results in",
    ],
    "adversative": [
        "however", "but", "yet", "nevertheless", "nonetheless",
        "although", "though", "despite", "in spite of", "conversely",
        "on the other hand", "in contrast", "while", "whereas",
        "even so", "still", "regardless",
    ],
    "additive": [
        "moreover", "furthermore", "additionally", "also", "besides",
        "in addition", "what's more", "similarly", "likewise",
        "equally", "too", "as well", "and",
    ],
    "temporal": [
        "then", "next", "subsequently", "afterward", "finally",
        "meanwhile", "previously", "earlier", "later", "first",
        "second", "third", "initially", "eventually", "ultimately",
    ],
    "explanatory": [
        "that is", "namely", "specifically", "in other words",
        "for example", "for instance", "such as", "to illustrate",
        "in particular", "particularly", "especially",
    ],
}


class SentenceClassifier:
    """Classifies sentences by rhetorical role and logical relation."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def classify_role(self, sentence: str, position: int, total: int) -> RhetoricalRole:
        """Classify the rhetorical role of a sentence.

        Uses semantic similarity to prototype phrases.
        """
        sentence_lower = sentence.lower()

        # Position-based heuristics
        if position == 0:
            # First sentence - likely claim or transition
            if self._has_evidence_markers(sentence_lower):
                return RhetoricalRole.EVIDENCE
            return RhetoricalRole.CLAIM

        if position == total - 1:
            # Last sentence - likely summary or conclusion
            if self._has_summary_markers(sentence_lower):
                return RhetoricalRole.SUMMARY
            return RhetoricalRole.CLAIM

        # Content-based classification
        if self._has_example_markers(sentence_lower):
            return RhetoricalRole.EXAMPLE

        if self._has_evidence_markers(sentence_lower):
            return RhetoricalRole.EVIDENCE

        if self._has_concession_markers(sentence_lower):
            return RhetoricalRole.CONCESSION

        if self._has_contrast_markers(sentence_lower):
            return RhetoricalRole.CONTRAST

        # Use semantic similarity for finer classification
        doc = self.nlp(sentence)
        role_scores = self._compute_role_similarities(doc)

        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            if role_scores[best_role] > 0.4:
                return best_role

        # Default to elaboration for middle sentences
        return RhetoricalRole.ELABORATION

    def classify_relation(self, sentence: str) -> LogicalRelation:
        """Classify the logical relation to the previous sentence."""
        sentence_lower = sentence.lower()
        words = sentence_lower.split()
        first_words = " ".join(words[:3]) if len(words) >= 3 else sentence_lower

        # Check for explicit connectors
        for relation_type, connectors in CONNECTOR_PATTERNS.items():
            for connector in connectors:
                if (first_words.startswith(connector) or
                    f" {connector} " in sentence_lower[:50]):
                    return LogicalRelation(relation_type)

        # Check for causal markers
        if any(m in sentence_lower for m in ["therefore", "thus", "hence", "so ", "because"]):
            return LogicalRelation.CAUSAL

        # Check for adversative markers
        if any(m in sentence_lower for m in ["however", "but ", "yet ", "although"]):
            return LogicalRelation.ADVERSATIVE

        # Check for additive markers
        if any(m in sentence_lower for m in ["moreover", "furthermore", "also", "additionally"]):
            return LogicalRelation.ADDITIVE

        return LogicalRelation.ADDITIVE  # Default

    def _has_example_markers(self, text: str) -> bool:
        markers = ["for example", "for instance", "such as", "e.g.", "consider"]
        return any(m in text for m in markers)

    def _has_evidence_markers(self, text: str) -> bool:
        markers = ["study", "research", "data", "evidence", "shows", "found", "according to", "percent", "%"]
        return any(m in text for m in markers)

    def _has_concession_markers(self, text: str) -> bool:
        markers = ["although", "while", "granted", "admittedly", "it is true", "certainly"]
        return any(m in text for m in markers)

    def _has_contrast_markers(self, text: str) -> bool:
        markers = ["however", "but", "yet", "on the other hand", "in contrast", "conversely"]
        return any(m in text for m in markers)

    def _has_summary_markers(self, text: str) -> bool:
        markers = ["in conclusion", "in summary", "to summarize", "overall", "ultimately", "in short"]
        return any(m in text for m in markers)

    def _compute_role_similarities(self, doc) -> Dict[RhetoricalRole, float]:
        """Compute similarity to prototype phrases for each role."""
        prototypes = {
            RhetoricalRole.CLAIM: "making an argument or stating a thesis",
            RhetoricalRole.EVIDENCE: "presenting data or research findings",
            RhetoricalRole.REASONING: "explaining logical connections",
            RhetoricalRole.EXAMPLE: "giving a concrete example or illustration",
            RhetoricalRole.CONTRAST: "contrasting with a different perspective",
        }

        scores = {}
        for role, prototype in prototypes.items():
            proto_doc = self.nlp(prototype)
            if doc.has_vector and proto_doc.has_vector:
                scores[role] = doc.similarity(proto_doc)

        return scores


class CorpusStatisticsExtractor:
    """Extracts comprehensive statistics from an author's corpus."""

    def __init__(self):
        self._nlp = None
        self.classifier = SentenceClassifier()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, paragraphs: List[str]) -> CorpusStatistics:
        """Extract statistics from a list of paragraphs.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            CorpusStatistics with all extracted metrics.
        """
        logger.info(f"Extracting statistics from {len(paragraphs)} paragraphs")

        all_sentences = []
        all_paragraph_sentences = []
        all_transitions = []
        word_counter = Counter()
        connector_counter = Counter()

        for para in paragraphs:
            sentences = split_into_sentences(para)
            if not sentences:
                continue

            all_sentences.extend(sentences)
            all_paragraph_sentences.append(sentences)

            # Count words and connectors
            for sent in sentences:
                self._count_words(sent, word_counter, connector_counter)

            # Extract transitions between consecutive sentences
            for i in range(len(sentences) - 1):
                from_role = self.classifier.classify_role(
                    sentences[i], i, len(sentences)
                )
                to_role = self.classifier.classify_role(
                    sentences[i + 1], i + 1, len(sentences)
                )
                all_transitions.append((from_role.value, to_role.value))

        if not all_sentences:
            logger.warning("No sentences found in corpus")
            return CorpusStatistics()

        # Compute sentence-level statistics
        sentence_lengths = [len(s.split()) for s in all_sentences]
        complexities = [self._compute_complexity(s) for s in all_sentences]

        # Compute paragraph-level statistics
        paragraph_lengths = [len(p) for p in all_paragraph_sentences]
        burstiness_values = [
            calculate_burstiness(p) for p in all_paragraph_sentences
            if len(p) >= 2
        ]

        # Extract rhythm patterns
        rhythm_patterns = self._extract_rhythm_patterns(all_paragraph_sentences)

        # Build transition matrix
        transition_matrix = self._build_transition_matrix(all_transitions)

        # Compute vocabulary distributions
        total_words = sum(word_counter.values())
        general_word_freq = {
            word: count / total_words
            for word, count in word_counter.most_common(500)
            if word.lower() not in FUNCTION_WORDS
        }

        total_connectors = sum(connector_counter.values())
        connector_freq = {
            word: count / total_connectors
            for word, count in connector_counter.items()
        } if total_connectors > 0 else {}

        # Compute paragraph feature vectors
        feature_mean, feature_std = self._compute_paragraph_features(
            all_paragraph_sentences
        )

        stats = CorpusStatistics(
            sentence_length_mean=np.mean(sentence_lengths),
            sentence_length_std=np.std(sentence_lengths),
            complexity_mean=np.mean(complexities),
            complexity_std=np.std(complexities),
            paragraph_length_mean=np.mean(paragraph_lengths) if paragraph_lengths else 5.0,
            paragraph_length_std=np.std(paragraph_lengths) if paragraph_lengths else 2.0,
            burstiness_mean=np.mean(burstiness_values) if burstiness_values else 0.3,
            burstiness_std=np.std(burstiness_values) if burstiness_values else 0.1,
            rhythm_patterns=rhythm_patterns,
            transition_matrix=transition_matrix,
            general_word_frequencies=general_word_freq,
            connector_frequencies=connector_freq,
            paragraph_feature_mean=feature_mean,
            paragraph_feature_std=feature_std,
        )

        logger.info(
            f"Extracted stats: mean sentence length={stats.sentence_length_mean:.1f}, "
            f"burstiness={stats.burstiness_mean:.2f}, "
            f"{len(rhythm_patterns)} rhythm patterns"
        )

        return stats

    def extract_vocabulary_profile(self, paragraphs: List[str]) -> VocabularyProfile:
        """Extract vocabulary profile from corpus.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            VocabularyProfile with word distributions.
        """
        general_words = Counter()
        connectors_by_type = defaultdict(Counter)
        verbs = Counter()
        modifiers = Counter()
        function_words = Counter()
        words_by_pos = defaultdict(Counter)

        for para in paragraphs:
            doc = self.nlp(para)

            for token in doc:
                word = token.text.lower()
                lemma = token.lemma_.lower()

                if token.is_punct or token.is_space:
                    continue

                # Classify by POS
                if token.pos_ == "VERB" and not token.is_stop:
                    verbs[lemma] += 1
                    words_by_pos["VERB"][lemma] += 1

                elif token.pos_ in ("ADJ", "ADV") and not token.is_stop:
                    modifiers[lemma] += 1
                    words_by_pos[token.pos_][lemma] += 1

                elif token.pos_ == "NOUN" and not token.is_stop:
                    words_by_pos["NOUN"][lemma] += 1
                    if len(lemma) > 2:
                        general_words[lemma] += 1

                # Track function words
                if word in FUNCTION_WORDS:
                    function_words[word] += 1

                # Track connectors
                for conn_type, conn_list in CONNECTOR_PATTERNS.items():
                    if word in conn_list or lemma in conn_list:
                        connectors_by_type[conn_type][word] += 1

        # Normalize frequencies
        total_general = sum(general_words.values()) or 1
        total_verbs = sum(verbs.values()) or 1
        total_modifiers = sum(modifiers.values()) or 1
        total_function = sum(function_words.values()) or 1

        return VocabularyProfile(
            general_words={w: c / total_general for w, c in general_words.most_common(200)},
            connectors={
                conn_type: [(w, c) for w, c in counter.most_common(10)]
                for conn_type, counter in connectors_by_type.items()
            },
            common_verbs={w: c / total_verbs for w, c in verbs.most_common(100)},
            modifiers={w: c / total_modifiers for w, c in modifiers.most_common(100)},
            function_words={w: c / total_function for w, c in function_words.most_common(50)},
            words_by_pos={
                pos: {w: c for w, c in counter.most_common(100)}
                for pos, counter in words_by_pos.items()
            },
        )

    def _count_words(
        self,
        sentence: str,
        word_counter: Counter,
        connector_counter: Counter
    ) -> None:
        """Count words and connectors in a sentence."""
        doc = self.nlp(sentence)

        for token in doc:
            if token.is_punct or token.is_space:
                continue

            word = token.text.lower()
            lemma = token.lemma_.lower()

            # Skip function words for general count
            if word not in FUNCTION_WORDS:
                word_counter[lemma] += 1

            # Count connectors
            for conn_list in CONNECTOR_PATTERNS.values():
                if word in conn_list or lemma in conn_list:
                    connector_counter[word] += 1
                    break

    def _compute_complexity(self, sentence: str) -> float:
        """Compute syntactic complexity as dependency tree depth."""
        doc = self.nlp(sentence)

        max_depth = 0
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 20:  # Safety limit
                    break
            max_depth = max(max_depth, depth)

        return float(max_depth)

    def _extract_rhythm_patterns(
        self,
        paragraph_sentences: List[List[str]],
        min_pattern_length: int = 3,
        max_patterns: int = 10
    ) -> List[List[float]]:
        """Extract common rhythm patterns (normalized length sequences).

        Args:
            paragraph_sentences: List of [sentences] for each paragraph.
            min_pattern_length: Minimum sentences for a pattern.
            max_patterns: Maximum patterns to return.

        Returns:
            List of normalized length patterns.
        """
        patterns = []

        for sentences in paragraph_sentences:
            if len(sentences) < min_pattern_length:
                continue

            lengths = [len(s.split()) for s in sentences]
            mean_length = np.mean(lengths)

            if mean_length > 0:
                # Normalize to relative lengths
                normalized = [l / mean_length for l in lengths]
                patterns.append(normalized)

        if not patterns:
            return [[1.0, 0.8, 1.2]]  # Default pattern

        # Cluster similar patterns and return representatives
        # For now, just return unique patterns (rounded)
        unique_patterns = []
        for pattern in patterns:
            rounded = [round(v, 1) for v in pattern]
            if rounded not in unique_patterns:
                unique_patterns.append(rounded)

        return unique_patterns[:max_patterns]

    def _build_transition_matrix(
        self,
        transitions: List[Tuple[str, str]]
    ) -> Dict[str, Dict[str, float]]:
        """Build Markov transition matrix from observed transitions.

        Args:
            transitions: List of (from_role, to_role) tuples.

        Returns:
            Dict of from_role -> {to_role -> probability}.
        """
        if not transitions:
            return {}

        counts = defaultdict(Counter)
        for from_role, to_role in transitions:
            counts[from_role][to_role] += 1

        matrix = {}
        for from_role, to_counts in counts.items():
            total = sum(to_counts.values())
            matrix[from_role] = {
                to_role: count / total
                for to_role, count in to_counts.items()
            }

        return matrix

    def _compute_paragraph_features(
        self,
        paragraph_sentences: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute feature vectors for paragraph similarity.

        Features:
        - Sentence count
        - Mean sentence length
        - Sentence length std
        - Burstiness
        - Mean complexity
        - First sentence length (relative)
        - Last sentence length (relative)

        Returns:
            Tuple of (feature_mean, feature_std).
        """
        features = []

        for sentences in paragraph_sentences:
            if len(sentences) < 2:
                continue

            lengths = [len(s.split()) for s in sentences]
            mean_length = np.mean(lengths)
            complexities = [self._compute_complexity(s) for s in sentences]

            feature_vec = [
                len(sentences),                      # Sentence count
                mean_length,                         # Mean sentence length
                np.std(lengths),                     # Length std
                calculate_burstiness(sentences),    # Burstiness
                np.mean(complexities),               # Mean complexity
                lengths[0] / mean_length if mean_length > 0 else 1.0,   # First relative
                lengths[-1] / mean_length if mean_length > 0 else 1.0,  # Last relative
            ]
            features.append(feature_vec)

        if not features:
            return np.zeros(7), np.ones(7)

        features_array = np.array(features)
        return np.mean(features_array, axis=0), np.std(features_array, axis=0)
