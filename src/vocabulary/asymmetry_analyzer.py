"""Asymmetry analyzer for evaluating organic complexity in text.

Measures patterns that distinguish human writing from AI:
- Periodic sentences (delayed main verb)
- Interruptive syntax (em-dash digressions)
- Inverted openings (prep/adverb first, not subject)
- Balanced structures (bad - "A and B" patterns)
- Subjective adjectives

Higher organic_score = more human-like.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from ..utils.nlp import get_nlp, split_into_sentences, calculate_burstiness
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AsymmetryStats:
    """Statistics from asymmetry analysis."""
    total_sentences: int = 0

    # Good patterns (higher = more organic)
    periodic_sentences: int = 0  # Main verb in latter half
    em_dash_interruptions: int = 0  # Mid-sentence digressions
    inverted_openings: int = 0  # Start with prep/adverb
    subjective_adjectives: int = 0  # Emotional adjectives

    # Bad patterns (lower = more organic)
    balanced_structures: int = 0  # "A and B" clauses
    topic_sentence_openings: int = 0  # Abstract summary openers
    logic_gates: int = 0  # "therefore", "thus", etc.

    # Derived metrics
    burstiness: float = 0.0  # Sentence length variance

    # Detailed findings
    flagged_sentences: List[str] = field(default_factory=list)
    good_examples: List[str] = field(default_factory=list)

    @property
    def organic_score(self) -> float:
        """Calculate overall organic complexity score (0-100)."""
        if self.total_sentences == 0:
            return 0.0

        # Good patterns (positive contribution)
        good_ratio = (
            self.periodic_sentences +
            self.em_dash_interruptions * 2 +  # Weight em-dashes higher
            self.inverted_openings
        ) / max(1, self.total_sentences)

        # Bad patterns (negative contribution)
        bad_ratio = (
            self.balanced_structures * 2 +  # Weight balanced structures higher
            self.topic_sentence_openings * 2 +
            self.logic_gates
        ) / max(1, self.total_sentences)

        # Burstiness contribution (0-0.3 typical range)
        burstiness_bonus = min(self.burstiness * 50, 20)  # Cap at 20 points

        # Calculate score
        score = (good_ratio * 50) - (bad_ratio * 30) + burstiness_bonus
        return max(0, min(100, score))  # Clamp to 0-100

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== ORGANIC COMPLEXITY ANALYSIS ===",
            f"",
            f"Total sentences: {self.total_sentences}",
            f"Organic score: {self.organic_score:.1f}/100",
            f"Burstiness (CV): {self.burstiness:.3f} (target > 0.4)",
            f"",
            f"GOOD PATTERNS:",
            f"  Periodic sentences: {self.periodic_sentences} ({100*self.periodic_sentences/max(1,self.total_sentences):.0f}%)",
            f"  Em-dash interruptions: {self.em_dash_interruptions}",
            f"  Inverted openings: {self.inverted_openings} ({100*self.inverted_openings/max(1,self.total_sentences):.0f}%)",
            f"",
            f"BAD PATTERNS:",
            f"  Balanced structures: {self.balanced_structures}",
            f"  Topic sentence openers: {self.topic_sentence_openings}",
            f"  Logic gates: {self.logic_gates}",
        ]

        if self.flagged_sentences:
            lines.append("")
            lines.append("FLAGGED ISSUES:")
            for flag in self.flagged_sentences[:5]:
                lines.append(f"  - {flag}")

        if self.good_examples:
            lines.append("")
            lines.append("GOOD EXAMPLES:")
            for ex in self.good_examples[:3]:
                lines.append(f"  + {ex[:80]}...")

        return "\n".join(lines)


class AsymmetryAnalyzer:
    """Analyze text for organic vs mechanical complexity."""

    # Inverted opening words (prep, adverb, conj)
    INVERTED_STARTERS = {
        'for', 'in', 'at', 'on', 'from', 'through', 'beneath', 'beyond', 'within',
        'then', 'so', 'yet', 'but', 'and', 'nor',
        'there', 'here', 'now', 'thus', 'thence', 'hence',
        'never', 'always', 'often', 'sometimes', 'perhaps',
        'slowly', 'suddenly', 'gradually', 'finally',
    }

    # Logic gate words (mechanical transitions)
    LOGIC_GATES = {
        'therefore', 'thus', 'hence', 'consequently', 'accordingly',
        'moreover', 'furthermore', 'additionally',
        'however', 'nevertheless', 'nonetheless',
    }

    # Topic sentence patterns (academic openers)
    TOPIC_PATTERNS = [
        r'^[A-Z][a-z]+ experience (?:supports|suggests|indicates|shows)',
        r'^It (?:is|was|will be) (?:important|worth|evident|clear|notable|necessary)',
        r'^The (?:concept|notion|idea|principle|theory) of',
        r'^(?:This|That|These|Those) (?:suggests?|indicates?|shows?|demonstrates?)',
        r'^In (?:this|the) (?:context|regard|respect|case)',
    ]

    # Subjective adjectives (emotional, not neutral)
    SUBJECTIVE_ADJECTIVES = {
        'hideous', 'horrible', 'terrible', 'dreadful', 'awful', 'ghastly',
        'wretched', 'accursed', 'damned', 'cursed', 'abominable',
        'eldritch', 'nameless', 'unspeakable', 'unnameable', 'blasphemous',
        'loathsome', 'monstrous', 'grotesque', 'cyclopean', 'titanic',
        'ancient', 'primordial', 'immemorial', 'aeons-old',
        'furtive', 'stealthy', 'sinister', 'ominous', 'foreboding',
    }

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        """Lazy load spaCy."""
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def analyze(self, text: str) -> AsymmetryStats:
        """Analyze text for organic complexity markers.

        Args:
            text: Text to analyze.

        Returns:
            AsymmetryStats with detailed metrics.
        """
        stats = AsymmetryStats()

        if not text or not text.strip():
            return stats

        doc = self.nlp(text)
        sentences = list(doc.sents)
        stats.total_sentences = len(sentences)

        # Calculate burstiness
        sent_texts = [sent.text for sent in sentences]
        stats.burstiness = calculate_burstiness(sent_texts)

        for sent in sentences:
            sent_text = sent.text.strip()
            tokens = list(sent)

            if len(tokens) < 5:
                continue

            # Check for periodic structure (main verb late)
            if self._is_periodic(tokens, sent):
                stats.periodic_sentences += 1
                if len(stats.good_examples) < 5:
                    stats.good_examples.append(f"[PERIODIC] {sent_text}")

            # Check for em-dash interruptions
            if '—' in sent_text or ' – ' in sent_text:
                stats.em_dash_interruptions += 1
                if len(stats.good_examples) < 5:
                    stats.good_examples.append(f"[EM-DASH] {sent_text}")

            # Check for inverted opening
            first_word = tokens[0].text.lower().rstrip(',')
            if first_word in self.INVERTED_STARTERS:
                stats.inverted_openings += 1

            # Check for balanced "A, and B" structure
            if self._has_balanced_and(sent_text):
                stats.balanced_structures += 1
                stats.flagged_sentences.append(f"[BALANCED] {sent_text[:60]}...")

            # Check for topic sentence patterns
            for pattern in self.TOPIC_PATTERNS:
                if re.match(pattern, sent_text, re.IGNORECASE):
                    stats.topic_sentence_openings += 1
                    stats.flagged_sentences.append(f"[TOPIC] {sent_text[:60]}...")
                    break

            # Check for logic gates
            for token in tokens[:5]:
                if token.text.lower() in self.LOGIC_GATES:
                    stats.logic_gates += 1
                    stats.flagged_sentences.append(f"[LOGIC] {sent_text[:60]}...")
                    break

            # Count subjective adjectives
            for token in tokens:
                if token.text.lower() in self.SUBJECTIVE_ADJECTIVES:
                    stats.subjective_adjectives += 1

        return stats

    def _is_periodic(self, tokens, sent) -> bool:
        """Check if main verb is in latter half of sentence.

        Periodic sentences build up context before the main point.
        """
        # Find the ROOT (main verb)
        root_tokens = [t for t in tokens if t.dep_ == "ROOT"]
        if not root_tokens:
            return False

        root = root_tokens[0]
        root_pos = root.i - sent.start

        # Periodic if root is in latter 60% of sentence
        return root_pos > len(tokens) * 0.4

    def _has_balanced_and(self, sent_text: str) -> bool:
        """Detect balanced 'A, and B' structures at sentence start.

        Pattern: "The X is Y, and the Z is W" (balanced, mechanical)
        """
        # Check for ", and " pattern in first 50 chars
        first_part = sent_text[:50]
        if ', and ' not in first_part:
            return False

        # Split at ", and"
        before_and = first_part.split(', and')[0]

        # If the part before "and" is short and starts with article/determiner,
        # it's likely a balanced structure
        words_before = before_and.split()
        if len(words_before) < 8:
            first_word = words_before[0].lower() if words_before else ''
            if first_word in ('the', 'a', 'an', 'this', 'that', 'it'):
                return True

        return False


# Module singleton
_analyzer: AsymmetryAnalyzer = None


def get_asymmetry_analyzer() -> AsymmetryAnalyzer:
    """Get or create singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AsymmetryAnalyzer()
    return _analyzer


def analyze_organic_complexity(text: str) -> AsymmetryStats:
    """Convenience function to analyze text."""
    analyzer = get_asymmetry_analyzer()
    return analyzer.analyze(text)
