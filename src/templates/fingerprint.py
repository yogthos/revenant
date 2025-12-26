"""Style fingerprint extraction and verification.

Captures author's statistical voice signature using established
stylometric techniques (Burrows' Delta, function words, n-grams)
and provides constraints for LLM generation with verification.

Based on:
- Burrows' Delta (Burrows 2002, Argamon 2008)
- Writeprints feature set (Abbasi & Chen 2008)
- Function word analysis (Mosteller & Wallace 1964)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Counter as CounterType
from collections import Counter
import numpy as np
import re

from ..utils.nlp import get_nlp, split_into_sentences, calculate_burstiness
from ..utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Common Function Words (style-indicative, content-independent)
# =============================================================================

# Top function words used in stylometry (Burrows' original list + extensions)
FUNCTION_WORDS = {
    # Articles
    "the", "a", "an",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    # Prepositions
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "than", "as", "if", "when", "while", "although",
    "because", "since", "unless", "until", "whether",
    # Auxiliary verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    # Common adverbs
    "not", "very", "just", "now", "then", "here", "there", "where",
    "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "any", "only", "own", "same",
    # Discourse markers
    "however", "therefore", "thus", "hence", "consequently", "moreover",
    "furthermore", "nevertheless", "nonetheless", "accordingly",
    "indeed", "certainly", "perhaps", "probably", "possibly",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StyleFingerprint:
    """Statistical fingerprint of an author's style."""

    # Function word frequencies (normalized)
    function_word_freq: Dict[str, float] = field(default_factory=dict)

    # Sentence-level statistics
    sentence_length_mean: float = 0.0
    sentence_length_std: float = 0.0
    sentence_length_min: int = 0
    sentence_length_max: int = 0

    # Paragraph-level statistics
    paragraph_length_mean: float = 0.0  # sentences per paragraph
    burstiness_mean: float = 0.0  # sentence length variation

    # Vocabulary richness
    type_token_ratio: float = 0.0  # unique words / total words
    hapax_ratio: float = 0.0  # words appearing once / unique words

    # POS tag distribution
    pos_distribution: Dict[str, float] = field(default_factory=dict)

    # N-gram frequencies (character trigrams for style)
    char_trigram_freq: Dict[str, float] = field(default_factory=dict)

    # Punctuation patterns
    punctuation_freq: Dict[str, float] = field(default_factory=dict)

    # Preferred discourse markers
    preferred_transitions: List[str] = field(default_factory=list)

    # Characteristic phrases (distinctive n-grams)
    characteristic_phrases: List[str] = field(default_factory=list)

    # Author's distinctive vocabulary (high TF-IDF)
    distinctive_vocabulary: Dict[str, float] = field(default_factory=dict)

    # Framing patterns (sentence starters)
    framing_patterns: List[str] = field(default_factory=list)


@dataclass
class StyleConstraints:
    """Constraints derived from fingerprint for LLM generation."""

    # Hard constraints
    target_sentence_length: Tuple[int, int]  # (min, max) words
    required_transitions: List[str]  # Must use these
    forbidden_words: Set[str]  # Author never uses these

    # Soft constraints (preferences)
    preferred_vocabulary: List[str]  # Ranked by distinctiveness
    preferred_pos_pattern: str  # e.g., "ADJ NOUN VERB"
    preferred_punctuation: Dict[str, float]  # Usage rates

    # Style guidance for prompts
    framing_examples: List[str]  # Example sentence starters
    characteristic_phrases: List[str]  # Phrases to use

    def to_prompt_string(self) -> str:
        """Convert constraints to prompt format."""
        lines = []

        lines.append(f"Sentence length: {self.target_sentence_length[0]}-{self.target_sentence_length[1]} words")

        if self.preferred_vocabulary:
            lines.append(f"Preferred vocabulary: {', '.join(self.preferred_vocabulary[:15])}")

        if self.required_transitions:
            lines.append(f"Transition words to use: {', '.join(self.required_transitions[:5])}")

        if self.framing_examples:
            lines.append(f"Example sentence starters: {'; '.join(self.framing_examples[:3])}")

        if self.characteristic_phrases:
            lines.append(f"Characteristic phrases: {'; '.join(self.characteristic_phrases[:5])}")

        return "\n".join(lines)


@dataclass
class StyleVerification:
    """Result of verifying generated text against style fingerprint."""

    is_acceptable: bool
    delta_score: float  # Burrows' Delta (lower = more similar)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Detailed metrics
    sentence_length_deviation: float = 0.0
    function_word_deviation: float = 0.0
    vocabulary_overlap: float = 0.0


# =============================================================================
# Style Fingerprint Extractor
# =============================================================================

class StyleFingerprintExtractor:
    """Extracts style fingerprint from author corpus."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, paragraphs: List[str]) -> StyleFingerprint:
        """Extract style fingerprint from corpus paragraphs.

        Args:
            paragraphs: List of author's paragraph texts.

        Returns:
            StyleFingerprint with all extracted features.
        """
        logger.info(f"Extracting style fingerprint from {len(paragraphs)} paragraphs")

        all_sentences = []
        all_tokens = []
        word_counter = Counter()
        pos_counter = Counter()
        char_trigram_counter = Counter()
        punct_counter = Counter()
        transition_counter = Counter()
        sentence_starters = Counter()

        for para in paragraphs:
            sentences = split_into_sentences(para)
            all_sentences.extend(sentences)

            for sent in sentences:
                doc = self.nlp(sent)

                # Count tokens and POS
                sent_tokens = []
                for token in doc:
                    if not token.is_space:
                        word = token.text.lower()
                        sent_tokens.append(word)
                        word_counter[word] += 1
                        pos_counter[token.pos_] += 1

                        if token.is_punct:
                            punct_counter[token.text] += 1

                all_tokens.extend(sent_tokens)

                # Track sentence starters
                if sent_tokens:
                    starter = " ".join(sent_tokens[:3])
                    sentence_starters[starter] += 1

                # Extract character trigrams
                sent_clean = re.sub(r'\s+', ' ', sent.lower())
                for i in range(len(sent_clean) - 2):
                    trigram = sent_clean[i:i+3]
                    char_trigram_counter[trigram] += 1

                # Track discourse markers
                for word in sent_tokens:
                    if word in FUNCTION_WORDS:
                        if word in {"however", "therefore", "thus", "hence",
                                   "consequently", "moreover", "furthermore",
                                   "nevertheless", "accordingly", "indeed"}:
                            transition_counter[word] += 1

        if not all_sentences:
            logger.warning("No sentences found in corpus")
            return StyleFingerprint()

        # Compute sentence statistics
        sentence_lengths = [len(s.split()) for s in all_sentences]

        # Compute paragraph statistics
        para_lengths = []
        burstiness_vals = []
        for para in paragraphs:
            sents = split_into_sentences(para)
            if sents:
                para_lengths.append(len(sents))
                if len(sents) >= 2:
                    burstiness_vals.append(calculate_burstiness(sents))

        # Compute function word frequencies
        total_words = sum(word_counter.values())
        function_word_freq = {
            word: word_counter[word] / total_words
            for word in FUNCTION_WORDS
            if word in word_counter
        }

        # Compute POS distribution
        total_pos = sum(pos_counter.values())
        pos_dist = {pos: count / total_pos for pos, count in pos_counter.items()}

        # Compute character trigram frequencies (top 100)
        total_trigrams = sum(char_trigram_counter.values())
        char_trigram_freq = {
            tri: count / total_trigrams
            for tri, count in char_trigram_counter.most_common(100)
        }

        # Compute punctuation frequencies
        total_punct = sum(punct_counter.values())
        punct_freq = {
            p: count / total_punct
            for p, count in punct_counter.items()
        } if total_punct > 0 else {}

        # Vocabulary richness
        unique_words = len(word_counter)
        hapax = sum(1 for w, c in word_counter.items() if c == 1)

        # Distinctive vocabulary (high frequency non-function words)
        distinctive = {
            word: count / total_words
            for word, count in word_counter.most_common(200)
            if word not in FUNCTION_WORDS and len(word) > 3
        }

        # Preferred transitions
        transitions = [w for w, _ in transition_counter.most_common(10)]

        # Framing patterns (common sentence starters)
        framing = [s for s, _ in sentence_starters.most_common(20)]

        # Characteristic phrases (frequent 3-grams)
        phrases = self._extract_characteristic_phrases(paragraphs)

        return StyleFingerprint(
            function_word_freq=function_word_freq,
            sentence_length_mean=np.mean(sentence_lengths),
            sentence_length_std=np.std(sentence_lengths),
            sentence_length_min=min(sentence_lengths),
            sentence_length_max=max(sentence_lengths),
            paragraph_length_mean=np.mean(para_lengths) if para_lengths else 0,
            burstiness_mean=np.mean(burstiness_vals) if burstiness_vals else 0,
            type_token_ratio=unique_words / total_words if total_words > 0 else 0,
            hapax_ratio=hapax / unique_words if unique_words > 0 else 0,
            pos_distribution=pos_dist,
            char_trigram_freq=char_trigram_freq,
            punctuation_freq=punct_freq,
            preferred_transitions=transitions,
            distinctive_vocabulary=distinctive,
            framing_patterns=framing[:10],
            characteristic_phrases=phrases[:20],
        )

    def _extract_characteristic_phrases(self, paragraphs: List[str]) -> List[str]:
        """Extract characteristic word n-grams."""
        trigram_counter = Counter()

        for para in paragraphs:
            doc = self.nlp(para)
            tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]

            for i in range(len(tokens) - 2):
                trigram = " ".join(tokens[i:i+3])
                # Filter out all-function-word trigrams
                words = trigram.split()
                if not all(w in FUNCTION_WORDS for w in words):
                    trigram_counter[trigram] += 1

        # Return phrases that appear multiple times
        return [phrase for phrase, count in trigram_counter.most_common(50) if count >= 2]

    def derive_constraints(
        self,
        fingerprint: StyleFingerprint,
        strictness: float = 0.5,
    ) -> StyleConstraints:
        """Derive generation constraints from fingerprint.

        Args:
            fingerprint: The extracted style fingerprint.
            strictness: 0.0 = loose, 1.0 = strict matching.

        Returns:
            StyleConstraints for LLM generation.
        """
        # Sentence length range
        slack = int((1 - strictness) * fingerprint.sentence_length_std * 2)
        min_len = max(5, int(fingerprint.sentence_length_mean - fingerprint.sentence_length_std - slack))
        max_len = int(fingerprint.sentence_length_mean + fingerprint.sentence_length_std + slack)

        # Required transitions (top ones based on strictness)
        n_transitions = max(3, int(len(fingerprint.preferred_transitions) * strictness))
        required_transitions = fingerprint.preferred_transitions[:n_transitions]

        # Preferred vocabulary (top distinctive words)
        n_vocab = max(20, int(100 * strictness))
        preferred_vocab = list(fingerprint.distinctive_vocabulary.keys())[:n_vocab]

        # Framing examples
        n_framing = max(3, int(len(fingerprint.framing_patterns) * strictness))
        framing = fingerprint.framing_patterns[:n_framing]

        # Characteristic phrases
        n_phrases = max(5, int(len(fingerprint.characteristic_phrases) * strictness))
        phrases = fingerprint.characteristic_phrases[:n_phrases]

        return StyleConstraints(
            target_sentence_length=(min_len, max_len),
            required_transitions=required_transitions,
            forbidden_words=set(),  # Could be populated with anti-patterns
            preferred_vocabulary=preferred_vocab,
            preferred_pos_pattern="",  # Could derive from POS distribution
            preferred_punctuation=fingerprint.punctuation_freq,
            framing_examples=framing,
            characteristic_phrases=phrases,
        )


# =============================================================================
# Style Verifier (Burrows' Delta)
# =============================================================================

class StyleVerifier:
    """Verifies generated text matches author's style using Burrows' Delta."""

    def __init__(self, fingerprint: StyleFingerprint, threshold: float = 1.5):
        """Initialize verifier.

        Args:
            fingerprint: Author's style fingerprint.
            threshold: Maximum acceptable Burrows' Delta (lower = stricter).
        """
        self.fingerprint = fingerprint
        self.threshold = threshold
        self._nlp = None

        # Pre-compute corpus statistics for Delta
        self._corpus_mean = {}
        self._corpus_std = {}
        self._prepare_delta_stats()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def _prepare_delta_stats(self):
        """Prepare corpus mean and std for Delta calculation."""
        # Use function word frequencies as the feature vector
        for word, freq in self.fingerprint.function_word_freq.items():
            self._corpus_mean[word] = freq
            # Approximate std as fraction of mean (would need full corpus for exact)
            self._corpus_std[word] = max(freq * 0.3, 0.001)

    def verify(self, text: str) -> StyleVerification:
        """Verify generated text against style fingerprint.

        Args:
            text: Generated text to verify.

        Returns:
            StyleVerification with delta score and issues.
        """
        issues = []
        suggestions = []

        # Calculate Burrows' Delta
        delta = self._calculate_burrows_delta(text)

        # Check sentence length
        sentences = split_into_sentences(text)
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            mean_len = np.mean(lengths)
            expected_mean = self.fingerprint.sentence_length_mean
            expected_std = self.fingerprint.sentence_length_std

            len_deviation = abs(mean_len - expected_mean) / max(expected_std, 1)
            if len_deviation > 2:
                issues.append(f"Sentence length off: {mean_len:.1f} vs expected {expected_mean:.1f}")
                suggestions.append(f"Target {int(expected_mean)} words per sentence")
        else:
            len_deviation = 0

        # Check function word usage
        func_deviation = self._check_function_words(text)
        if func_deviation > 0.3:
            issues.append(f"Function word distribution differs significantly")

        # Check vocabulary overlap
        vocab_overlap = self._check_vocabulary_overlap(text)
        if vocab_overlap < 0.1:
            issues.append("Low vocabulary overlap with author's style")
            top_vocab = list(self.fingerprint.distinctive_vocabulary.keys())[:10]
            suggestions.append(f"Use vocabulary like: {', '.join(top_vocab)}")

        # Check transitions
        if self.fingerprint.preferred_transitions:
            text_lower = text.lower()
            used_transitions = [t for t in self.fingerprint.preferred_transitions if t in text_lower]
            if not used_transitions:
                issues.append("Missing characteristic transition words")
                suggestions.append(f"Use transitions: {', '.join(self.fingerprint.preferred_transitions[:5])}")

        is_acceptable = delta < self.threshold and len(issues) <= 2

        return StyleVerification(
            is_acceptable=is_acceptable,
            delta_score=delta,
            issues=issues,
            suggestions=suggestions,
            sentence_length_deviation=len_deviation,
            function_word_deviation=func_deviation,
            vocabulary_overlap=vocab_overlap,
        )

    def _calculate_burrows_delta(self, text: str) -> float:
        """Calculate Burrows' Delta between text and corpus.

        Delta = mean of |z-score(text) - z-score(corpus)| for each feature.

        Args:
            text: Text to compare.

        Returns:
            Delta score (lower = more similar).
        """
        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc if not t.is_space]
        total = len(tokens)

        if total == 0:
            return 10.0  # Maximum dissimilarity

        # Calculate frequencies in text
        text_freq = Counter(tokens)
        text_freq = {w: c / total for w, c in text_freq.items()}

        # Calculate z-scores and differences
        deltas = []
        for word in self._corpus_mean:
            corpus_z = (self._corpus_mean[word] - self._corpus_mean[word]) / self._corpus_std[word]  # = 0
            text_z = (text_freq.get(word, 0) - self._corpus_mean[word]) / self._corpus_std[word]
            deltas.append(abs(text_z - corpus_z))

        return np.mean(deltas) if deltas else 10.0

    def _check_function_words(self, text: str) -> float:
        """Check function word distribution similarity.

        Returns:
            Deviation score (0 = identical, higher = more different).
        """
        doc = self.nlp(text)
        tokens = [t.text.lower() for t in doc if not t.is_space]
        total = len(tokens)

        if total == 0:
            return 1.0

        text_freq = Counter(tokens)

        # Compare top function words
        deviations = []
        for word, corpus_freq in self.fingerprint.function_word_freq.items():
            text_freq_val = text_freq.get(word, 0) / total
            if corpus_freq > 0:
                deviation = abs(text_freq_val - corpus_freq) / corpus_freq
                deviations.append(deviation)

        return np.mean(deviations) if deviations else 1.0

    def _check_vocabulary_overlap(self, text: str) -> float:
        """Check overlap with author's distinctive vocabulary.

        Returns:
            Overlap ratio (0 = no overlap, 1 = full overlap).
        """
        doc = self.nlp(text)
        text_words = {t.lemma_.lower() for t in doc if not t.is_space and not t.is_punct}

        distinctive = set(self.fingerprint.distinctive_vocabulary.keys())

        if not distinctive:
            return 0.5  # No baseline

        overlap = len(text_words & distinctive)
        return overlap / len(distinctive)

    def suggest_repairs(self, text: str, verification: StyleVerification) -> List[str]:
        """Suggest specific repairs for style issues.

        Args:
            text: The text that failed verification.
            verification: The verification result.

        Returns:
            List of repair suggestions.
        """
        repairs = []

        if verification.sentence_length_deviation > 2:
            target = self.fingerprint.sentence_length_mean
            repairs.append(f"Adjust sentence length to ~{int(target)} words")

        if verification.function_word_deviation > 0.3:
            # Find underused function words
            doc = self.nlp(text)
            tokens = [t.text.lower() for t in doc if not t.is_space]
            total = len(tokens)
            text_freq = Counter(tokens)

            for word, corpus_freq in sorted(
                self.fingerprint.function_word_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]:
                text_freq_val = text_freq.get(word, 0) / max(total, 1)
                if text_freq_val < corpus_freq * 0.5:
                    repairs.append(f"Increase use of '{word}'")

        if verification.vocabulary_overlap < 0.1:
            top_words = list(self.fingerprint.distinctive_vocabulary.keys())[:10]
            repairs.append(f"Incorporate vocabulary: {', '.join(top_words)}")

        # Add transition suggestions
        if any("transition" in issue.lower() for issue in verification.issues):
            repairs.append(
                f"Add transitions: {', '.join(self.fingerprint.preferred_transitions[:5])}"
            )

        return repairs
