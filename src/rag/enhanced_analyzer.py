"""Enhanced structural analyzer for concrete style patterns.

Implements 5-channel analysis to address:
- Mechanical Precision → Syntactic Templates (concrete POS patterns)
- Impersonal Tone → Vocabulary Clusters + Emotional Stance Markers
- Sophisticated Clarity → Transition Inventory + Opening Patterns

Key insight: Abstract guidance ("use long sentences") doesn't help.
Concrete patterns ("DET ADJ NOUN — ADV VERB") do.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import re

from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Common words to exclude from vocabulary extraction
COMMON_ADVERBS = {
    'very', 'really', 'quite', 'just', 'also', 'only', 'even', 'still',
    'already', 'always', 'never', 'often', 'usually', 'sometimes',
    'now', 'then', 'here', 'there', 'too', 'so', 'well', 'much',
}

COMMON_ADJECTIVES = {
    'good', 'great', 'new', 'old', 'big', 'small', 'long', 'short',
    'high', 'low', 'first', 'last', 'next', 'other', 'same', 'different',
    'many', 'few', 'more', 'most', 'own', 'able', 'certain', 'sure',
}


@dataclass
class SyntacticTemplate:
    """Concrete syntactic template from author's prose."""
    pos_pattern: str          # "DET ADJ NOUN — ADV VERB PREP DET NOUN"
    clause_structure: str     # "main + subordinate" or "fragment"
    length_category: str      # "LONG", "SHORT", etc.
    example_skeleton: str     # "The [ADJ] [NOUN] — [ADV] [VERB] upon the [NOUN]"
    frequency: float = 1.0    # How often this pattern appears

    def to_instruction(self) -> str:
        """Format as prompt instruction."""
        return f'"{self.pos_pattern}"\n  Skeleton: "{self.example_skeleton}"'


@dataclass
class VocabularyCluster:
    """Vocabulary grouped by stylistic function."""
    intensifiers: List[str] = field(default_factory=list)   # "utterly", "tremendously"
    evaluatives: List[str] = field(default_factory=list)    # "blasphemous", "magnificent"
    emotional: List[str] = field(default_factory=list)      # "dread", "horror", "wonder"
    sensory: List[str] = field(default_factory=list)        # "fetid", "writhing", "luminous"
    archaic: List[str] = field(default_factory=list)        # "whereupon", "whilst"
    stance_certain: List[str] = field(default_factory=list) # "clearly", "obviously"
    stance_hedge: List[str] = field(default_factory=list)   # "perhaps", "possibly"

    def to_instruction(self) -> str:
        """Format as prompt instruction."""
        lines = []
        if self.intensifiers:
            lines.append(f"Intensifiers: {', '.join(self.intensifiers[:5])}")
        if self.evaluatives:
            lines.append(f"Evaluatives: {', '.join(self.evaluatives[:5])}")
        if self.emotional:
            lines.append(f"Emotional: {', '.join(self.emotional[:5])}")
        if self.archaic:
            lines.append(f"Archaic: {', '.join(self.archaic[:5])}")
        return "\n".join(lines) if lines else ""


@dataclass
class TransitionInventory:
    """Author's transition vocabulary."""
    additive: List[str] = field(default_factory=list)      # "and", "moreover"
    adversative: List[str] = field(default_factory=list)   # "yet", "but", "however"
    causal: List[str] = field(default_factory=list)        # "thus", "therefore"
    temporal: List[str] = field(default_factory=list)      # "then", "whereupon"
    exemplifying: List[str] = field(default_factory=list)  # "indeed", "specifically"
    avoid: List[str] = field(default_factory=list)         # Words NOT in author's corpus

    def to_instruction(self) -> str:
        """Format as prompt instruction."""
        lines = []
        if self.adversative:
            lines.append(f"Adversative: {', '.join(self.adversative[:4])}")
        if self.causal:
            lines.append(f"Causal: {', '.join(self.causal[:4])}")
        if self.temporal:
            lines.append(f"Temporal: {', '.join(self.temporal[:4])}")
        if self.avoid:
            lines.append(f"AVOID: {', '.join(self.avoid[:4])}")
        return "\n".join(lines) if lines else ""


@dataclass
class StanceProfile:
    """Author's stance and emotional engagement patterns."""
    certainty_markers: List[str] = field(default_factory=list)
    hedging_markers: List[str] = field(default_factory=list)
    rhetorical_question_freq: float = 0.0
    exclamation_freq: float = 0.0
    direct_address_freq: float = 0.0
    parenthetical_freq: float = 0.0

    def to_instruction(self) -> str:
        """Format as prompt instruction."""
        lines = []
        if self.certainty_markers:
            lines.append(f"Certainty markers: {', '.join(self.certainty_markers[:4])}")
        if self.rhetorical_question_freq > 0.05:
            lines.append(f"Rhetorical questions: ~{int(self.rhetorical_question_freq * 100)}% of paragraphs")
        if self.exclamation_freq > 0.02:
            lines.append("Use occasional exclamations for emphasis")
        if self.parenthetical_freq > 0.1:
            lines.append("Use parenthetical asides (em-dashes)")
        return "\n".join(lines) if lines else ""


@dataclass
class OpeningPatterns:
    """Sentence opening patterns with frequencies."""
    patterns: Dict[str, float] = field(default_factory=dict)
    avoid_patterns: List[str] = field(default_factory=list)

    def to_instruction(self) -> str:
        """Format as prompt instruction."""
        lines = []
        sorted_patterns = sorted(self.patterns.items(), key=lambda x: -x[1])
        for pattern, freq in sorted_patterns[:4]:
            pct = int(freq * 100)
            if pct >= 5:
                lines.append(f"{pct}% - {pattern}")
        if self.avoid_patterns:
            lines.append(f"AVOID: {', '.join(self.avoid_patterns[:3])}")
        return "\n".join(lines) if lines else ""


@dataclass
class EnhancedStyleProfile:
    """Complete enhanced style profile for prompt injection."""
    syntactic_templates: List[SyntacticTemplate] = field(default_factory=list)
    vocabulary: VocabularyCluster = field(default_factory=VocabularyCluster)
    transitions: TransitionInventory = field(default_factory=TransitionInventory)
    stance: StanceProfile = field(default_factory=StanceProfile)
    openings: OpeningPatterns = field(default_factory=OpeningPatterns)

    def format_for_prompt(self) -> str:
        """Format complete enhanced guidance for prompt."""
        sections = []

        # Syntactic templates
        if self.syntactic_templates:
            templates = "\n".join(f"  • {t.to_instruction()}" for t in self.syntactic_templates[:3])
            sections.append(f"SYNTACTIC TEMPLATES:\n{templates}")

        # Vocabulary
        vocab_instr = self.vocabulary.to_instruction()
        if vocab_instr:
            sections.append(f"VOCABULARY:\n  {vocab_instr.replace(chr(10), chr(10) + '  ')}")

        # Transitions
        trans_instr = self.transitions.to_instruction()
        if trans_instr:
            sections.append(f"TRANSITIONS:\n  {trans_instr.replace(chr(10), chr(10) + '  ')}")

        # Emotional engagement
        stance_instr = self.stance.to_instruction()
        if stance_instr:
            sections.append(f"EMOTIONAL ENGAGEMENT:\n  {stance_instr.replace(chr(10), chr(10) + '  ')}")

        # Openings
        opening_instr = self.openings.to_instruction()
        if opening_instr:
            sections.append(f"SENTENCE OPENINGS:\n  {opening_instr.replace(chr(10), chr(10) + '  ')}")

        return "\n\n".join(sections) if sections else ""


class EnhancedStructuralAnalyzer:
    """Multi-channel analyzer for concrete style patterns.

    Extracts:
    1. Syntactic Templates - concrete POS patterns with skeletons
    2. Vocabulary Clusters - author-specific word banks by function
    3. Transition Inventory - specific connectives the author uses
    4. Emotional Stance Markers - attitude and engagement patterns
    5. Opening Patterns - sentence-initial POS patterns
    """

    # Length categories (word counts)
    LENGTH_THRESHOLDS = {
        "FRAGMENT": (1, 4),
        "SHORT": (5, 10),
        "MEDIUM": (11, 20),
        "LONG": (21, 35),
        "VERY_LONG": (36, 1000),
    }

    # Transition categories
    ADDITIVE = {'and', 'moreover', 'furthermore', 'also', 'besides', 'additionally', 'plus'}
    ADVERSATIVE = {'but', 'yet', 'however', 'nevertheless', 'nonetheless', 'still', 'though', 'although'}
    CAUSAL = {'thus', 'therefore', 'hence', 'consequently', 'wherefore', 'so', 'accordingly'}
    TEMPORAL = {'then', 'whereupon', 'thereafter', 'meanwhile', 'subsequently', 'afterward', 'next'}
    EXEMPLIFYING = {'indeed', 'specifically', 'namely', 'particularly', 'especially'}

    # LLM-speak transitions to avoid
    LLM_TRANSITIONS = {'furthermore', 'additionally', 'moreover', 'in addition', 'it is important'}

    # Certainty and hedging markers
    CERTAINTY = {'clearly', 'obviously', 'undeniably', 'surely', 'certainly', 'undoubtedly', 'plainly'}
    HEDGES = {'perhaps', 'possibly', 'maybe', 'seemingly', 'apparently', 'arguably', 'conceivably'}

    # Archaic vocabulary
    ARCHAIC_WORDS = {
        'whereupon', 'whilst', 'heretofore', 'hitherto', 'thereof', 'wherein',
        'thereby', 'forthwith', 'henceforth', 'thence', 'whence', 'whereof',
        'notwithstanding', 'erstwhile', 'forsooth', 'verily', 'betwixt',
    }

    # Emotional/evaluative lexicon seeds
    EMOTIONAL_NOUNS = {
        'dread', 'horror', 'terror', 'fear', 'awe', 'wonder', 'fascination',
        'revulsion', 'loathing', 'despair', 'anguish', 'melancholy', 'rapture',
        'ecstasy', 'triumph', 'fury', 'rage', 'passion', 'yearning', 'longing',
    }

    SENSORY_ADJECTIVES = {
        'fetid', 'putrid', 'luminous', 'phosphorescent', 'writhing', 'pulsating',
        'viscous', 'gelatinous', 'squamous', 'rugose', 'cyclopean', 'eldritch',
        'gibbous', 'tenebrous', 'crepuscular', 'iridescent', 'opalescent',
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

    def _analyze_clause_structure(self, sent) -> str:
        """Analyze clause structure of a sentence."""
        # Count subordinating conjunctions and relative pronouns
        subordinators = sum(1 for t in sent if t.dep_ in {'mark', 'relcl', 'advcl', 'ccomp'})
        has_fragment = not any(t.pos_ == "VERB" and t.dep_ == "ROOT" for t in sent)

        if has_fragment:
            return "fragment"
        elif subordinators == 0:
            return "main"
        elif subordinators == 1:
            return "main + subordinate"
        else:
            return "complex"

    def _create_skeleton(self, sent) -> str:
        """Create fillable skeleton from sentence."""
        parts = []
        for token in sent:
            if token.is_punct:
                parts.append(token.text)
            elif token.is_space:
                continue
            elif token.pos_ in {'DET', 'ADP', 'CCONJ', 'SCONJ', 'PART', 'PRON'}:
                # Keep function words
                parts.append(token.text.lower())
            else:
                # Replace content words with POS placeholders
                parts.append(f"[{token.pos_}]")
        return " ".join(parts)

    # ==========================================================================
    # Phase 1: Syntactic Template Extraction
    # ==========================================================================

    def extract_syntactic_templates(self, texts: List[str]) -> List[SyntacticTemplate]:
        """Extract POS templates with clause structure."""
        template_counter: Counter = Counter()
        templates_by_pattern: Dict[str, SyntacticTemplate] = {}

        for text in texts:
            doc = self.nlp(text)
            for sent in doc.sents:
                # Extract POS sequence with punctuation preserved
                pos_tokens = []
                for token in sent:
                    if token.is_punct:
                        pos_tokens.append(token.text)
                    elif not token.is_space:
                        pos_tokens.append(token.pos_)

                if not pos_tokens:
                    continue

                pos_pattern = " ".join(pos_tokens)
                clause_structure = self._analyze_clause_structure(sent)
                word_count = len([t for t in sent if not t.is_punct and not t.is_space])
                skeleton = self._create_skeleton(sent)

                # Simplify pattern to first 10 elements for grouping
                simplified = " ".join(pos_tokens[:10])
                template_counter[simplified] += 1

                if simplified not in templates_by_pattern:
                    templates_by_pattern[simplified] = SyntacticTemplate(
                        pos_pattern=pos_pattern[:100],  # Truncate very long patterns
                        clause_structure=clause_structure,
                        length_category=self.categorize_length(word_count),
                        example_skeleton=skeleton[:100],
                    )

        # Return most common templates
        total = sum(template_counter.values()) or 1
        result = []
        for pattern, count in template_counter.most_common(10):
            template = templates_by_pattern[pattern]
            template.frequency = count / total
            result.append(template)

        return result

    # ==========================================================================
    # Phase 2: Vocabulary Cluster Extraction
    # ==========================================================================

    def extract_vocabulary_clusters(self, texts: List[str]) -> VocabularyCluster:
        """Extract distinctive vocabulary by function."""
        intensifiers: Counter = Counter()
        evaluatives: Counter = Counter()
        emotional: Counter = Counter()
        sensory: Counter = Counter()
        archaic: Counter = Counter()
        stance_certain: Counter = Counter()
        stance_hedge: Counter = Counter()

        for text in texts:
            doc = self.nlp(text)
            for token in doc:
                word_lower = token.text.lower()

                # Intensifiers: adverbs modifying adjectives
                if token.pos_ == "ADV" and token.head.pos_ == "ADJ":
                    if word_lower not in COMMON_ADVERBS:
                        intensifiers[word_lower] += 1

                # Evaluatives: strong adjectives
                if token.pos_ == "ADJ":
                    if word_lower not in COMMON_ADJECTIVES:
                        # Check if it's unusual/distinctive
                        if len(word_lower) > 5 or word_lower in self.SENSORY_ADJECTIVES:
                            evaluatives[word_lower] += 1

                # Emotional nouns
                if token.pos_ == "NOUN" and word_lower in self.EMOTIONAL_NOUNS:
                    emotional[word_lower] += 1

                # Sensory adjectives
                if token.pos_ == "ADJ" and word_lower in self.SENSORY_ADJECTIVES:
                    sensory[word_lower] += 1

                # Archaic vocabulary
                if word_lower in self.ARCHAIC_WORDS:
                    archaic[word_lower] += 1

                # Stance markers
                if word_lower in self.CERTAINTY:
                    stance_certain[word_lower] += 1
                elif word_lower in self.HEDGES:
                    stance_hedge[word_lower] += 1

        return VocabularyCluster(
            intensifiers=[w for w, _ in intensifiers.most_common(10)],
            evaluatives=[w for w, _ in evaluatives.most_common(15)],
            emotional=[w for w, _ in emotional.most_common(10)],
            sensory=[w for w, _ in sensory.most_common(10)],
            archaic=[w for w, _ in archaic.most_common(10)],
            stance_certain=[w for w, _ in stance_certain.most_common(5)],
            stance_hedge=[w for w, _ in stance_hedge.most_common(5)],
        )

    # ==========================================================================
    # Phase 3: Transition Inventory Extraction
    # ==========================================================================

    def extract_transitions(self, texts: List[str]) -> TransitionInventory:
        """Extract transition words and their frequencies."""
        transitions: Dict[str, Counter] = {
            'additive': Counter(),
            'adversative': Counter(),
            'causal': Counter(),
            'temporal': Counter(),
            'exemplifying': Counter(),
        }
        author_transitions: Set[str] = set()

        for text in texts:
            doc = self.nlp(text)
            for sent in doc.sents:
                # Get first word (skip punctuation/space)
                first_word = None
                for token in sent:
                    if not token.is_space and not token.is_punct:
                        first_word = token.text.lower()
                        break

                if first_word:
                    author_transitions.add(first_word)

                    if first_word in self.ADDITIVE:
                        transitions['additive'][first_word] += 1
                    elif first_word in self.ADVERSATIVE:
                        transitions['adversative'][first_word] += 1
                    elif first_word in self.CAUSAL:
                        transitions['causal'][first_word] += 1
                    elif first_word in self.TEMPORAL:
                        transitions['temporal'][first_word] += 1
                    elif first_word in self.EXEMPLIFYING:
                        transitions['exemplifying'][first_word] += 1

        # Find LLM-speak transitions NOT in author's corpus
        avoid = [t for t in self.LLM_TRANSITIONS if t not in author_transitions]

        return TransitionInventory(
            additive=[w for w, _ in transitions['additive'].most_common(5)],
            adversative=[w for w, _ in transitions['adversative'].most_common(5)],
            causal=[w for w, _ in transitions['causal'].most_common(5)],
            temporal=[w for w, _ in transitions['temporal'].most_common(5)],
            exemplifying=[w for w, _ in transitions['exemplifying'].most_common(5)],
            avoid=avoid,
        )

    # ==========================================================================
    # Phase 4: Emotional Stance Markers Extraction
    # ==========================================================================

    def extract_stance_profile(self, texts: List[str]) -> StanceProfile:
        """Analyze author's emotional engagement and stance."""
        certainty: Counter = Counter()
        hedges: Counter = Counter()
        rhetorical_qs = 0
        exclamations = 0
        direct_address = 0
        parentheticals = 0
        total_sents = 0
        total_paragraphs = len(texts)

        for text in texts:
            doc = self.nlp(text)
            for sent in doc.sents:
                total_sents += 1
                sent_text = sent.text

                # Check for rhetorical questions
                if sent_text.strip().endswith('?'):
                    rhetorical_qs += 1

                # Check for exclamations
                if '!' in sent_text:
                    exclamations += 1

                # Check for parentheticals (em-dashes, parentheses)
                if '—' in sent_text or '(' in sent_text:
                    parentheticals += 1

                # Check for stance markers and direct address
                for token in sent:
                    word = token.text.lower()
                    if word in self.CERTAINTY:
                        certainty[word] += 1
                    elif word in self.HEDGES:
                        hedges[word] += 1

                    # Direct address pronouns
                    if word in {'you', 'your', 'we', 'our', 'one'}:
                        direct_address += 1

        total_sents = max(1, total_sents)
        total_paragraphs = max(1, total_paragraphs)

        return StanceProfile(
            certainty_markers=[w for w, _ in certainty.most_common(5)],
            hedging_markers=[w for w, _ in hedges.most_common(5)],
            rhetorical_question_freq=rhetorical_qs / total_paragraphs,
            exclamation_freq=exclamations / total_sents,
            direct_address_freq=direct_address / total_sents,
            parenthetical_freq=parentheticals / total_sents,
        )

    # ==========================================================================
    # Phase 5: Opening Pattern Extraction
    # ==========================================================================

    def extract_opening_patterns(self, texts: List[str]) -> OpeningPatterns:
        """Extract sentence-initial POS patterns."""
        patterns: Counter = Counter()
        total = 0

        for text in texts:
            doc = self.nlp(text)
            for sent in doc.sents:
                # Get first 3 POS tags
                pos_tags = []
                for token in sent:
                    if token.is_space:
                        continue
                    if len(pos_tags) >= 3:
                        break
                    if token.is_punct:
                        pos_tags.append(token.text)
                    else:
                        pos_tags.append(token.pos_)

                if pos_tags:
                    pattern = " ".join(pos_tags)
                    patterns[pattern] += 1
                    total += 1

        # Convert to frequencies
        total = max(1, total)
        pattern_freq = {p: c / total for p, c in patterns.most_common(15)}

        return OpeningPatterns(
            patterns=pattern_freq,
            avoid_patterns=["Furthermore ,", "Additionally ,", "Moreover ,", "It is important"],
        )

    # ==========================================================================
    # Main Analysis Entry Point
    # ==========================================================================

    def analyze(self, texts: List[str]) -> EnhancedStyleProfile:
        """Run complete multi-channel analysis on corpus texts.

        Args:
            texts: List of text chunks (paragraphs) from author corpus.

        Returns:
            EnhancedStyleProfile with all extracted patterns.
        """
        if not texts:
            return EnhancedStyleProfile()

        logger.info(f"Running enhanced structural analysis on {len(texts)} chunks")

        return EnhancedStyleProfile(
            syntactic_templates=self.extract_syntactic_templates(texts),
            vocabulary=self.extract_vocabulary_clusters(texts),
            transitions=self.extract_transitions(texts),
            stance=self.extract_stance_profile(texts),
            openings=self.extract_opening_patterns(texts),
        )


# Module singleton
_enhanced_analyzer: Optional[EnhancedStructuralAnalyzer] = None


def get_enhanced_analyzer() -> EnhancedStructuralAnalyzer:
    """Get the enhanced structural analyzer singleton."""
    global _enhanced_analyzer
    if _enhanced_analyzer is None:
        _enhanced_analyzer = EnhancedStructuralAnalyzer()
    return _enhanced_analyzer
