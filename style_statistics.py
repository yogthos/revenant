"""
Style Statistics Module

Builds a statistical profile of the sample text and uses it to validate
generated text. Sentences that deviate too far from the statistical norms
of the sample are flagged for rejection/rewriting.

Key metrics:
- Sentence length distribution (mean, std, quartiles)
- Word length distribution
- POS tag frequencies
- Sentence opener patterns
- Punctuation usage patterns
- N-gram frequencies
"""

import re
import math
import spacy
import textstat
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np


@dataclass
class SentenceStats:
    """Statistics for a single sentence."""
    text: str
    word_count: int
    char_count: int
    avg_word_length: float
    syllable_count: int
    flesch_reading_ease: float
    opener_word: str
    opener_pos: str
    has_subordinate_clause: bool
    punctuation_count: Dict[str, int]
    pos_distribution: Dict[str, int]


@dataclass
class StyleProfile:
    """Statistical profile of a text's style."""
    # Sentence length
    sentence_length_mean: float
    sentence_length_std: float
    sentence_length_quartiles: Tuple[float, float, float]  # Q1, Q2, Q3

    # Word characteristics
    avg_word_length_mean: float
    avg_word_length_std: float
    syllables_per_word_mean: float

    # Readability
    flesch_reading_ease_mean: float
    flesch_reading_ease_std: float

    # Sentence openers
    opener_word_freq: Dict[str, float]  # Normalized frequencies
    opener_pos_freq: Dict[str, float]

    # POS tag distribution
    pos_tag_freq: Dict[str, float]

    # Punctuation patterns
    punctuation_per_sentence: Dict[str, float]

    # Discourse markers
    discourse_marker_freq: Dict[str, float]

    # N-grams (bigrams and trigrams)
    bigram_freq: Dict[str, float]
    trigram_freq: Dict[str, float]


class StyleStatisticsAnalyzer:
    """
    Analyzes text to build statistical style profiles and validates
    generated text against those profiles.
    """

    # Common discourse markers to track
    DISCOURSE_MARKERS = [
        'therefore', 'hence', 'thus', 'consequently', 'accordingly',
        'however', 'nevertheless', 'nonetheless', 'yet', 'but',
        'moreover', 'furthermore', 'additionally', 'also',
        'contrary to', 'in contrast', 'on the other hand',
        'for example', 'for instance', 'such as',
        'in this connection', 'it follows that', 'this means that',
        'clearly', 'evidently', 'obviously', 'plainly',
    ]

    def __init__(self):
        """Initialize the analyzer."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self._sample_profile: Optional[StyleProfile] = None

    def analyze_text(self, text: str) -> StyleProfile:
        """
        Build a complete statistical profile of the text.

        Args:
            text: The text to analyze

        Returns:
            StyleProfile with all statistical measures
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)

        # Analyze each sentence
        sentence_stats = []
        for sent in sentences:
            stats = self._analyze_sentence(sent)
            if stats.word_count >= 3:  # Skip very short sentences
                sentence_stats.append(stats)

        if not sentence_stats:
            raise ValueError("No valid sentences found in text")

        # Aggregate statistics
        lengths = [s.word_count for s in sentence_stats]
        avg_word_lens = [s.avg_word_length for s in sentence_stats]
        flesch_scores = [s.flesch_reading_ease for s in sentence_stats]
        syllables = [s.syllable_count / max(s.word_count, 1) for s in sentence_stats]

        # Sentence length quartiles
        q1, q2, q3 = np.percentile(lengths, [25, 50, 75])

        # Opener frequencies
        opener_word_counts = Counter(s.opener_word.lower() for s in sentence_stats)
        opener_pos_counts = Counter(s.opener_pos for s in sentence_stats)
        total_sents = len(sentence_stats)

        # POS tag aggregation
        all_pos = Counter()
        for s in sentence_stats:
            all_pos.update(s.pos_distribution)
        total_pos = sum(all_pos.values())

        # Punctuation aggregation
        all_punct = defaultdict(int)
        for s in sentence_stats:
            for punct, count in s.punctuation_count.items():
                all_punct[punct] += count

        # Discourse marker frequencies
        text_lower = text.lower()
        word_count = len(text.split())
        discourse_freq = {}
        for marker in self.DISCOURSE_MARKERS:
            count = len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
            discourse_freq[marker] = count / max(word_count, 1) * 1000  # Per 1000 words

        # N-gram frequencies
        bigrams = self._extract_ngrams(text, 2)
        trigrams = self._extract_ngrams(text, 3)

        return StyleProfile(
            sentence_length_mean=np.mean(lengths),
            sentence_length_std=np.std(lengths),
            sentence_length_quartiles=(q1, q2, q3),
            avg_word_length_mean=np.mean(avg_word_lens),
            avg_word_length_std=np.std(avg_word_lens),
            syllables_per_word_mean=np.mean(syllables),
            flesch_reading_ease_mean=np.mean(flesch_scores),
            flesch_reading_ease_std=np.std(flesch_scores),
            opener_word_freq={k: v / total_sents for k, v in opener_word_counts.most_common(50)},
            opener_pos_freq={k: v / total_sents for k, v in opener_pos_counts.items()},
            pos_tag_freq={k: v / total_pos for k, v in all_pos.items()},
            punctuation_per_sentence={k: v / total_sents for k, v in all_punct.items()},
            discourse_marker_freq=discourse_freq,
            bigram_freq=bigrams,
            trigram_freq=trigrams,
        )

    def _analyze_sentence(self, sent) -> SentenceStats:
        """Analyze a single sentence."""
        text = sent.text.strip()
        tokens = [t for t in sent if not t.is_space]
        words = [t for t in tokens if t.is_alpha]

        # Basic counts
        word_count = len(words)
        char_count = sum(len(t.text) for t in words)
        avg_word_length = char_count / max(word_count, 1)

        # Syllables and readability
        syllable_count = textstat.syllable_count(text)
        try:
            flesch = textstat.flesch_reading_ease(text)
        except:
            flesch = 50.0  # Default middle score

        # Opener analysis
        opener_word = words[0].text if words else ""
        opener_pos = words[0].pos_ if words else ""

        # Check for subordinate clauses
        has_subordinate = any(t.dep_ in ('mark', 'advcl', 'relcl') for t in tokens)

        # Punctuation
        punct_count = Counter(t.text for t in tokens if t.is_punct)

        # POS distribution
        pos_dist = Counter(t.pos_ for t in words)

        return SentenceStats(
            text=text,
            word_count=word_count,
            char_count=char_count,
            avg_word_length=avg_word_length,
            syllable_count=syllable_count,
            flesch_reading_ease=flesch,
            opener_word=opener_word,
            opener_pos=opener_pos,
            has_subordinate_clause=has_subordinate,
            punctuation_count=dict(punct_count),
            pos_distribution=dict(pos_dist),
        )

    def _extract_ngrams(self, text: str, n: int) -> Dict[str, float]:
        """Extract n-gram frequencies from text."""
        doc = self.nlp(text.lower())
        words = [t.text for t in doc if t.is_alpha and not t.is_stop]

        ngrams = Counter()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams[ngram] += 1

        # Normalize by total
        total = sum(ngrams.values())
        if total == 0:
            return {}

        # Return top 100 most common
        return {k: v / total for k, v in ngrams.most_common(100)}

    def set_sample_profile(self, sample_text: str):
        """
        Set the reference sample profile.

        Args:
            sample_text: The sample text to use as reference
        """
        self._sample_profile = self.analyze_text(sample_text)
        print(f"  [StyleStats] Sample profile built:")
        print(f"    - Sentence length: {self._sample_profile.sentence_length_mean:.1f} ± {self._sample_profile.sentence_length_std:.1f} words")
        print(f"    - Flesch score: {self._sample_profile.flesch_reading_ease_mean:.1f} ± {self._sample_profile.flesch_reading_ease_std:.1f}")

    def score_sentence(self, sentence: str) -> Tuple[float, List[str]]:
        """
        Score a sentence against the sample profile.

        Args:
            sentence: The sentence to score

        Returns:
            Tuple of (score 0-1, list of issues)
        """
        if not self._sample_profile:
            return 1.0, []

        profile = self._sample_profile
        doc = self.nlp(sentence)
        stats = self._analyze_sentence(doc[:])  # Treat whole doc as one sentence

        issues = []
        scores = []

        # 1. Sentence length score
        length_z = (stats.word_count - profile.sentence_length_mean) / max(profile.sentence_length_std, 1)
        length_score = max(0, 1 - abs(length_z) / 3)  # Penalize >3 std deviations
        scores.append(length_score)
        if abs(length_z) > 2:
            if length_z > 0:
                issues.append(f"Sentence too long ({stats.word_count} words, expected ~{profile.sentence_length_mean:.0f})")
            else:
                issues.append(f"Sentence too short ({stats.word_count} words, expected ~{profile.sentence_length_mean:.0f})")

        # 2. Word length score
        word_len_z = (stats.avg_word_length - profile.avg_word_length_mean) / max(profile.avg_word_length_std, 0.5)
        word_len_score = max(0, 1 - abs(word_len_z) / 3)
        scores.append(word_len_score)

        # 3. Readability score
        flesch_z = (stats.flesch_reading_ease - profile.flesch_reading_ease_mean) / max(profile.flesch_reading_ease_std, 10)
        flesch_score = max(0, 1 - abs(flesch_z) / 3)
        scores.append(flesch_score)
        if abs(flesch_z) > 2:
            issues.append(f"Readability differs significantly (Flesch: {stats.flesch_reading_ease:.0f}, expected ~{profile.flesch_reading_ease_mean:.0f})")

        # 4. Opener score (is this opener common in sample?)
        opener = stats.opener_word.lower()
        opener_freq = profile.opener_word_freq.get(opener, 0)
        # Common openers in sample should score higher
        opener_score = min(1.0, opener_freq * 10)  # Scale up frequency
        scores.append(opener_score * 0.5 + 0.5)  # Dampen - don't penalize too harshly

        # 5. Discourse marker overuse check
        sentence_lower = sentence.lower()
        word_count = len(sentence.split())
        for marker in self.DISCOURSE_MARKERS:
            if marker in sentence_lower:
                expected_freq = profile.discourse_marker_freq.get(marker, 0)
                # If this marker is rare in sample but appears in sentence, penalize
                if expected_freq < 1.0:  # Less than 1 per 1000 words in sample
                    issues.append(f"Rare discourse marker used: '{marker}' (sample uses sparingly)")

        # Calculate overall score
        overall_score = sum(scores) / len(scores)

        return overall_score, issues

    def score_text(self, text: str) -> Tuple[float, Dict[str, any]]:
        """
        Score an entire text against the sample profile.

        Args:
            text: The text to score

        Returns:
            Tuple of (score 0-1, detailed metrics dict)
        """
        if not self._sample_profile:
            return 1.0, {}

        # Analyze the generated text
        gen_profile = self.analyze_text(text)
        sample = self._sample_profile

        metrics = {}
        scores = []

        # 1. Sentence length distribution match
        len_diff = abs(gen_profile.sentence_length_mean - sample.sentence_length_mean)
        len_score = max(0, 1 - len_diff / sample.sentence_length_mean)
        metrics['sentence_length_match'] = len_score
        scores.append(len_score)

        # 2. Readability match
        flesch_diff = abs(gen_profile.flesch_reading_ease_mean - sample.flesch_reading_ease_mean)
        flesch_score = max(0, 1 - flesch_diff / 50)  # 50-point range is significant
        metrics['readability_match'] = flesch_score
        scores.append(flesch_score)

        # 3. Discourse marker usage match
        marker_scores = []
        overused_markers = []
        for marker, sample_freq in sample.discourse_marker_freq.items():
            gen_freq = gen_profile.discourse_marker_freq.get(marker, 0)
            if sample_freq > 0:
                ratio = gen_freq / sample_freq
                if ratio > 3:  # More than 3x overuse
                    overused_markers.append((marker, ratio))
                marker_scores.append(max(0, 1 - abs(1 - ratio) / 2))

        if marker_scores:
            marker_match = sum(marker_scores) / len(marker_scores)
        else:
            marker_match = 1.0
        metrics['discourse_marker_match'] = marker_match
        metrics['overused_markers'] = overused_markers
        scores.append(marker_match)

        # 4. POS distribution match (Jensen-Shannon divergence)
        pos_score = self._distribution_similarity(
            gen_profile.pos_tag_freq,
            sample.pos_tag_freq
        )
        metrics['pos_distribution_match'] = pos_score
        scores.append(pos_score)

        # 5. Opener diversity match
        gen_openers = set(gen_profile.opener_word_freq.keys())
        sample_openers = set(sample.opener_word_freq.keys())
        opener_overlap = len(gen_openers & sample_openers) / max(len(sample_openers), 1)
        metrics['opener_diversity_match'] = opener_overlap
        scores.append(opener_overlap)

        overall_score = sum(scores) / len(scores)
        metrics['overall_score'] = overall_score

        return overall_score, metrics

    def _distribution_similarity(self, dist1: Dict[str, float],
                                  dist2: Dict[str, float]) -> float:
        """Calculate similarity between two distributions using overlap."""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 1.0

        total_overlap = 0
        for key in all_keys:
            v1 = dist1.get(key, 0)
            v2 = dist2.get(key, 0)
            total_overlap += min(v1, v2)

        return total_overlap

    def get_rejection_sentences(self, text: str, threshold: float = 0.4) -> List[Tuple[str, float, List[str]]]:
        """
        Get sentences that should be rejected/rewritten.

        Args:
            text: The text to analyze
            threshold: Score below which to reject

        Returns:
            List of (sentence, score, issues) tuples
        """
        doc = self.nlp(text)
        rejections = []

        for sent in doc.sents:
            sentence = sent.text.strip()
            if len(sentence.split()) < 3:
                continue

            score, issues = self.score_sentence(sentence)
            if score < threshold:
                rejections.append((sentence, score, issues))

        return rejections


# Test function
if __name__ == '__main__':
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample.txt"
    if sample_path.exists():
        sample_text = sample_path.read_text()

        print("=== Style Statistics Test ===\n")

        analyzer = StyleStatisticsAnalyzer()
        analyzer.set_sample_profile(sample_text)

        print(f"\nSample discourse marker frequencies (per 1000 words):")
        for marker, freq in sorted(analyzer._sample_profile.discourse_marker_freq.items(),
                                   key=lambda x: -x[1])[:10]:
            print(f"  {marker}: {freq:.2f}")

        # Test sentence
        test_sentences = [
            "Consequently, this is a test sentence with discourse markers.",
            "Therefore, we must acknowledge that this implies a variety of factors.",
            "The material conditions of production determine the social relations.",
            "Contrary to metaphysics, dialectics holds that nature is in constant motion.",
        ]

        print("\n=== Sentence Scores ===")
        for sent in test_sentences:
            score, issues = analyzer.score_sentence(sent)
            print(f"\n'{sent[:60]}...'")
            print(f"  Score: {score:.2f}")
            if issues:
                for issue in issues:
                    print(f"  Issue: {issue}")
    else:
        print("No sample.txt found.")

