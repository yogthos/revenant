"""Structural style analyzer - extracts rhythm and syntax patterns without content.

Addresses issues with LoRA output:
- Mechanical Transitions → Extract natural transition patterns
- Impersonal Tone → Capture subjective/emotional markers
- Mechanical Precision → Inject sentence length variation
- Formulaic Flow → Extract rhythm fingerprints
- Robotic Formality → Capture informal punctuation patterns
"""

from dataclasses import dataclass, field
from typing import List
from collections import Counter

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SentencePattern:
    """Pattern extracted from a single sentence."""
    length: int  # Word count
    length_category: str  # SHORT, MEDIUM, LONG, VERY_LONG
    has_dash: bool
    has_semicolon: bool
    has_ellipsis: bool
    has_exclamation: bool
    has_question: bool
    opening_pos: str  # First word's POS tag
    is_fragment: bool  # No main verb
    has_parenthetical: bool
    clause_count: int  # Number of clauses


@dataclass
class RhythmFingerprint:
    """Rhythm pattern for a paragraph."""
    sentence_count: int
    length_sequence: List[str]  # ["LONG", "SHORT", "MEDIUM", ...]
    length_variance: float
    avg_sentence_length: float
    punctuation_density: float  # Special punctuation per sentence
    fragment_ratio: float  # Ratio of fragments to full sentences

    def to_rhythm_string(self) -> str:
        """Convert to human-readable rhythm instruction."""
        lengths = " → ".join(self.length_sequence[:6])  # First 6 sentences
        return f"[{lengths}]"

    def to_instruction(self) -> str:
        """Generate rhythm instruction for prompt."""
        parts = []

        # Length pattern
        if self.length_sequence:
            pattern = ", ".join(self.length_sequence[:5])
            parts.append(f"Sentence rhythm: {pattern}")

        # Variation guidance
        if self.length_variance > 100:
            parts.append("HIGH variation in sentence length")
        elif self.length_variance > 50:
            parts.append("MODERATE variation in sentence length")

        # Fragment usage
        if self.fragment_ratio > 0.2:
            parts.append(f"Use sentence fragments (~{int(self.fragment_ratio*100)}% of sentences)")

        # Punctuation
        if self.punctuation_density > 0.5:
            parts.append("Use dashes and semicolons liberally")

        return "; ".join(parts) if parts else ""


@dataclass
class SyntaxTemplate:
    """Abstract syntactic template without content words."""
    template: str  # e.g., "DET NOUN VERB . FRAGMENT . DET ADJ NOUN VERB PREP DET NOUN ;"
    pos_sequence: List[str]
    punctuation_pattern: str  # e.g., ". . ; —"

    def to_instruction(self) -> str:
        """Generate syntax instruction."""
        # Simplify to key structural markers
        markers = []
        if "—" in self.punctuation_pattern:
            markers.append("use dashes for interruption")
        if ";" in self.punctuation_pattern:
            markers.append("use semicolons to link clauses")
        if self.punctuation_pattern.count(".") > 2:
            markers.append("break into short sentences")
        return ", ".join(markers) if markers else ""


@dataclass
class StructuralStyle:
    """Complete structural style profile for injection."""
    rhythm: RhythmFingerprint
    templates: List[SyntaxTemplate] = field(default_factory=list)
    transition_words: List[str] = field(default_factory=list)
    opening_patterns: List[str] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format structural guidance for prompt injection."""
        lines = []

        # Rhythm instruction
        rhythm_instr = self.rhythm.to_instruction()
        if rhythm_instr:
            lines.append(f"RHYTHM: {rhythm_instr}")

        # Opening patterns (abstract)
        if self.opening_patterns:
            patterns = ", ".join(self.opening_patterns[:3])
            lines.append(f"OPENINGS: {patterns}")

        return "\n".join(lines) if lines else ""


class StructuralAnalyzer:
    """Analyzes text for structural patterns without extracting content."""

    # Length categories (word counts)
    LENGTH_THRESHOLDS = {
        "FRAGMENT": (1, 4),
        "SHORT": (5, 10),
        "MEDIUM": (11, 20),
        "LONG": (21, 35),
        "VERY_LONG": (36, 1000),
    }

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def categorize_length(self, word_count: int) -> str:
        """Categorize sentence length."""
        for category, (min_len, max_len) in self.LENGTH_THRESHOLDS.items():
            if min_len <= word_count <= max_len:
                return category
        return "VERY_LONG"

    def analyze_sentence(self, sent) -> SentencePattern:
        """Extract structural pattern from a spaCy sentence."""
        text = sent.text
        words = [t for t in sent if not t.is_punct and not t.is_space]
        word_count = len(words)

        # Check for fragment (no root verb)
        has_verb = any(t.pos_ == "VERB" and t.dep_ == "ROOT" for t in sent)
        is_fragment = not has_verb and word_count < 8

        # Opening POS
        opening_pos = ""
        for token in sent:
            if not token.is_space and not token.is_punct:
                opening_pos = token.pos_
                break

        # Count clauses (approximate by counting verbs)
        verb_count = sum(1 for t in sent if t.pos_ == "VERB")
        clause_count = max(1, verb_count)

        return SentencePattern(
            length=word_count,
            length_category=self.categorize_length(word_count),
            has_dash="—" in text or "–" in text or " - " in text,
            has_semicolon=";" in text,
            has_ellipsis="..." in text or "…" in text,
            has_exclamation="!" in text,
            has_question="?" in text,
            opening_pos=opening_pos,
            is_fragment=is_fragment,
            has_parenthetical="(" in text or "," in text.split()[1:5] if len(text.split()) > 5 else False,
            clause_count=clause_count,
        )

    def extract_rhythm(self, text: str) -> RhythmFingerprint:
        """Extract rhythm fingerprint from text."""
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return RhythmFingerprint(
                sentence_count=0,
                length_sequence=[],
                length_variance=0,
                avg_sentence_length=0,
                punctuation_density=0,
                fragment_ratio=0,
            )

        patterns = [self.analyze_sentence(s) for s in sentences]

        # Length sequence
        length_sequence = [p.length_category for p in patterns]
        lengths = [p.length for p in patterns]

        # Calculate variance
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths) if lengths else 0

        # Punctuation density
        special_punct = sum(1 for p in patterns if p.has_dash or p.has_semicolon or p.has_ellipsis)
        punct_density = special_punct / len(patterns) if patterns else 0

        # Fragment ratio
        fragments = sum(1 for p in patterns if p.is_fragment)
        fragment_ratio = fragments / len(patterns) if patterns else 0

        return RhythmFingerprint(
            sentence_count=len(sentences),
            length_sequence=length_sequence,
            length_variance=variance,
            avg_sentence_length=avg_len,
            punctuation_density=punct_density,
            fragment_ratio=fragment_ratio,
        )

    def extract_pos_template(self, text: str) -> SyntaxTemplate:
        """Extract abstract POS template from text."""
        doc = self.nlp(text)

        pos_tags = []
        punct_pattern = []

        for token in doc:
            if token.is_punct:
                punct_pattern.append(token.text)
            elif not token.is_space:
                # Use POS tag instead of actual word
                pos_tags.append(token.pos_)

        template = " ".join(pos_tags)
        punct_str = " ".join(punct_pattern)

        return SyntaxTemplate(
            template=template,
            pos_sequence=pos_tags,
            punctuation_pattern=punct_str,
        )

    def extract_opening_patterns(self, text: str, n: int = 5) -> List[str]:
        """Extract abstract opening patterns (first 2-3 words as POS)."""
        doc = self.nlp(text)
        patterns = []

        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_space][:3]
            if tokens:
                pattern = " ".join(t.pos_ for t in tokens)
                patterns.append(pattern)

        # Return most common patterns
        counter = Counter(patterns)
        return [p for p, _ in counter.most_common(n)]

    def analyze(self, text: str) -> StructuralStyle:
        """Full structural analysis of text."""
        rhythm = self.extract_rhythm(text)
        template = self.extract_pos_template(text)
        openings = self.extract_opening_patterns(text)

        return StructuralStyle(
            rhythm=rhythm,
            templates=[template],
            opening_patterns=openings,
        )


# Module singleton
_analyzer = None


def get_structural_analyzer() -> StructuralAnalyzer:
    """Get the structural analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = StructuralAnalyzer()
    return _analyzer


def extract_rhythm_instruction(text: str) -> str:
    """Convenience function to get rhythm instruction from text."""
    analyzer = get_structural_analyzer()
    rhythm = analyzer.extract_rhythm(text)
    return rhythm.to_instruction()
