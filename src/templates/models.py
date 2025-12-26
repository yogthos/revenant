"""Data models for template-based style transfer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


# =============================================================================
# Enums for classification dimensions
# =============================================================================

class SlotType(Enum):
    """Types of slots in a sentence template."""
    SUBJECT = "subject"           # Main subject noun phrase
    VERB = "verb"                 # Main verb phrase
    OBJECT = "object"             # Direct/indirect objects
    MODIFIER = "modifier"         # Adjectives, adverbs
    CLAUSE = "clause"             # Subordinate clauses
    CONNECTOR = "connector"       # Transition words
    PREPOSITIONAL = "prepositional"  # Prepositional phrases
    COMPLEMENT = "complement"     # Subject/object complements


class SentenceType(Enum):
    """Position-based sentence type in a paragraph."""
    OPENER = "opener"             # First sentence of paragraph
    BODY = "body"                 # Middle sentences
    CLOSER = "closer"             # Last sentence of paragraph


class RhetoricalRole(Enum):
    """Rhetorical function of a sentence."""
    CLAIM = "claim"               # Main argument/thesis statement
    EVIDENCE = "evidence"         # Supporting evidence or data
    REASONING = "reasoning"       # Logical reasoning connecting evidence to claim
    CONCESSION = "concession"     # Acknowledging counter-arguments
    CONTRAST = "contrast"         # Contrasting with previous point
    ELABORATION = "elaboration"   # Expanding on previous point
    EXAMPLE = "example"           # Concrete example or illustration
    TRANSITION = "transition"     # Transitional sentence
    SUMMARY = "summary"           # Summarizing previous points


class LogicalRelation(Enum):
    """Logical relationship to previous sentence."""
    NONE = "none"                 # No explicit relation (first sentence)
    CAUSAL = "causal"             # Cause-effect relationship
    ADVERSATIVE = "adversative"   # Contrast/opposition
    ADDITIVE = "additive"         # Adding information
    TEMPORAL = "temporal"         # Time sequence
    CONDITIONAL = "conditional"   # If-then relationship
    EXPLANATORY = "explanatory"   # Explaining previous point


class WordType(Enum):
    """Classification of word types for vocabulary management."""
    TECHNICAL = "technical"       # Domain-specific terms (preserve exactly)
    GENERAL = "general"           # General words (substitute with author's)
    FUNCTION = "function"         # Function words (the, a, of, etc.)
    CONNECTOR = "connector"       # Transition/connector words
    PROPER = "proper"             # Proper nouns (preserve exactly)


# =============================================================================
# Template Models
# =============================================================================

@dataclass
class TemplateSlot:
    """A slot in a sentence template that can be filled with content."""
    name: str                     # Unique identifier for the slot
    slot_type: SlotType           # Type of content expected
    position: int                 # Character position in skeleton
    required: bool = True         # Whether slot must be filled

    # Constraints
    pos_tags: List[str] = field(default_factory=list)  # Allowed POS tags
    min_words: int = 1
    max_words: int = 10

    # For CLAUSE slots
    clause_template: Optional["SentenceTemplate"] = None

    # For CONNECTOR slots
    connector_type: Optional[str] = None  # causal, adversative, etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "slot_type": self.slot_type.value,
            "position": self.position,
            "required": self.required,
            "pos_tags": self.pos_tags,
            "min_words": self.min_words,
            "max_words": self.max_words,
            "connector_type": self.connector_type,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TemplateSlot":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            slot_type=SlotType(data["slot_type"]),
            position=data["position"],
            required=data.get("required", True),
            pos_tags=data.get("pos_tags", []),
            min_words=data.get("min_words", 1),
            max_words=data.get("max_words", 10),
            connector_type=data.get("connector_type"),
        )


@dataclass
class SentenceTemplate:
    """A syntactic template extracted from an author's corpus."""
    id: str

    # The skeleton with placeholders: "The [SUBJECT] [VERB] the [OBJECT]."
    skeleton: str

    # POS pattern: "DET NOUN VERB DET NOUN PUNCT"
    pos_pattern: str

    # Structural metrics
    word_count: int
    complexity_score: float       # Dependency tree depth
    clause_count: int

    # Classification dimensions
    sentence_type: SentenceType
    rhetorical_role: RhetoricalRole
    logical_relation: LogicalRelation

    # Original text for reference
    original_text: str

    # Slot definitions
    slots: List[TemplateSlot] = field(default_factory=list)

    # Author metadata
    author: str = ""
    document_id: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "skeleton": self.skeleton,
            "pos_pattern": self.pos_pattern,
            "word_count": self.word_count,
            "complexity_score": self.complexity_score,
            "clause_count": self.clause_count,
            "sentence_type": self.sentence_type.value,
            "rhetorical_role": self.rhetorical_role.value,
            "logical_relation": self.logical_relation.value,
            "original_text": self.original_text,
            "slots": [s.to_dict() for s in self.slots],
            "author": self.author,
            "document_id": self.document_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SentenceTemplate":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            skeleton=data["skeleton"],
            pos_pattern=data["pos_pattern"],
            word_count=data["word_count"],
            complexity_score=data["complexity_score"],
            clause_count=data["clause_count"],
            sentence_type=SentenceType(data["sentence_type"]),
            rhetorical_role=RhetoricalRole(data["rhetorical_role"]),
            logical_relation=LogicalRelation(data["logical_relation"]),
            original_text=data["original_text"],
            slots=[TemplateSlot.from_dict(s) for s in data.get("slots", [])],
            author=data.get("author", ""),
            document_id=data.get("document_id", ""),
        )

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata format (flat structure)."""
        return {
            "author": self.author,
            "sentence_type": self.sentence_type.value,
            "rhetorical_role": self.rhetorical_role.value,
            "logical_relation": self.logical_relation.value,
            "word_count": self.word_count,
            "complexity_score": self.complexity_score,
            "clause_count": self.clause_count,
            "pos_pattern": self.pos_pattern,
            "skeleton": self.skeleton,
            "slot_count": len(self.slots),
            "document_id": self.document_id,
        }


@dataclass
class SentenceRequirements:
    """Requirements for selecting a sentence template."""
    sentence_type: SentenceType
    rhetorical_role: RhetoricalRole
    logical_relation: LogicalRelation = LogicalRelation.NONE

    # Length constraints
    min_length: int = 5
    max_length: int = 40
    target_length: Optional[int] = None

    # Complexity constraints
    min_complexity: float = 0.0
    max_complexity: float = 10.0
    target_complexity: Optional[float] = None


# =============================================================================
# Vocabulary Models
# =============================================================================

@dataclass
class VocabularyProfile:
    """Author's vocabulary profile for word substitution."""

    # General-purpose words (high frequency, non-technical)
    # word -> frequency (normalized)
    general_words: Dict[str, float] = field(default_factory=dict)

    # Connectors by type: causal -> ["therefore", "thus", ...]
    connectors: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    # Common verbs: verb -> frequency
    common_verbs: Dict[str, float] = field(default_factory=dict)

    # Modifiers (adjectives/adverbs): word -> frequency
    modifiers: Dict[str, float] = field(default_factory=dict)

    # Function words distribution: word -> frequency
    function_words: Dict[str, float] = field(default_factory=dict)

    # Words by POS tag: POS -> {word -> frequency}
    words_by_pos: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "general_words": self.general_words,
            "connectors": {
                k: [(w, f) for w, f in v]
                for k, v in self.connectors.items()
            },
            "common_verbs": self.common_verbs,
            "modifiers": self.modifiers,
            "function_words": self.function_words,
            "words_by_pos": self.words_by_pos,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "VocabularyProfile":
        """Create from dictionary."""
        return cls(
            general_words=data.get("general_words", {}),
            connectors={
                k: [(w, f) for w, f in v]
                for k, v in data.get("connectors", {}).items()
            },
            common_verbs=data.get("common_verbs", {}),
            modifiers=data.get("modifiers", {}),
            function_words=data.get("function_words", {}),
            words_by_pos=data.get("words_by_pos", {}),
        )

    def get_connector(self, connector_type: str, exclude: List[str] = None) -> Optional[str]:
        """Sample a connector of the given type, weighted by frequency."""
        exclude = exclude or []
        candidates = self.connectors.get(connector_type, [])
        candidates = [(w, f) for w, f in candidates if w not in exclude]

        if not candidates:
            return None

        words, freqs = zip(*candidates)
        total = sum(freqs)
        probs = [f / total for f in freqs]

        import random
        return random.choices(words, weights=probs, k=1)[0]

    def get_word_by_pos(self, pos: str, semantic_target: str = None) -> Optional[str]:
        """Get a word of the given POS, optionally similar to semantic target."""
        candidates = self.words_by_pos.get(pos, {})
        if not candidates:
            return None

        # If no semantic target, sample by frequency
        if not semantic_target:
            words, freqs = zip(*candidates.items())
            total = sum(freqs)
            probs = [f / total for f in freqs]
            import random
            return random.choices(words, weights=probs, k=1)[0]

        # TODO: Use semantic similarity to find best match
        # For now, return highest frequency word
        return max(candidates.items(), key=lambda x: x[1])[0]


# =============================================================================
# Corpus Statistics Models
# =============================================================================

@dataclass
class CorpusStatistics:
    """Comprehensive statistics extracted from an author's corpus."""

    # Sentence-level statistics
    sentence_length_mean: float = 15.0
    sentence_length_std: float = 5.0
    complexity_mean: float = 3.0
    complexity_std: float = 1.0

    # Paragraph-level statistics
    paragraph_length_mean: float = 5.0  # in sentences
    paragraph_length_std: float = 2.0
    burstiness_mean: float = 0.3
    burstiness_std: float = 0.1

    # Rhythm patterns (normalized length sequences)
    # e.g., [[1.0, 0.6, 1.4], [0.8, 1.2, 0.7, 1.3]]
    rhythm_patterns: List[List[float]] = field(default_factory=list)

    # Transition matrix: from_type -> to_type -> probability
    transition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Vocabulary distributions
    general_word_frequencies: Dict[str, float] = field(default_factory=dict)
    connector_frequencies: Dict[str, float] = field(default_factory=dict)

    # Feature vectors for paragraph similarity
    paragraph_feature_mean: Optional[np.ndarray] = None
    paragraph_feature_std: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "sentence_length_mean": self.sentence_length_mean,
            "sentence_length_std": self.sentence_length_std,
            "complexity_mean": self.complexity_mean,
            "complexity_std": self.complexity_std,
            "paragraph_length_mean": self.paragraph_length_mean,
            "paragraph_length_std": self.paragraph_length_std,
            "burstiness_mean": self.burstiness_mean,
            "burstiness_std": self.burstiness_std,
            "rhythm_patterns": self.rhythm_patterns,
            "transition_matrix": self.transition_matrix,
            "general_word_frequencies": self.general_word_frequencies,
            "connector_frequencies": self.connector_frequencies,
            "paragraph_feature_mean": (
                self.paragraph_feature_mean.tolist()
                if self.paragraph_feature_mean is not None else None
            ),
            "paragraph_feature_std": (
                self.paragraph_feature_std.tolist()
                if self.paragraph_feature_std is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CorpusStatistics":
        """Create from dictionary."""
        return cls(
            sentence_length_mean=data.get("sentence_length_mean", 15.0),
            sentence_length_std=data.get("sentence_length_std", 5.0),
            complexity_mean=data.get("complexity_mean", 3.0),
            complexity_std=data.get("complexity_std", 1.0),
            paragraph_length_mean=data.get("paragraph_length_mean", 5.0),
            paragraph_length_std=data.get("paragraph_length_std", 2.0),
            burstiness_mean=data.get("burstiness_mean", 0.3),
            burstiness_std=data.get("burstiness_std", 0.1),
            rhythm_patterns=data.get("rhythm_patterns", []),
            transition_matrix=data.get("transition_matrix", {}),
            general_word_frequencies=data.get("general_word_frequencies", {}),
            connector_frequencies=data.get("connector_frequencies", {}),
            paragraph_feature_mean=(
                np.array(data["paragraph_feature_mean"])
                if data.get("paragraph_feature_mean") else None
            ),
            paragraph_feature_std=(
                np.array(data["paragraph_feature_std"])
                if data.get("paragraph_feature_std") else None
            ),
        )

    def get_transition_probability(
        self,
        from_role: str,
        to_role: str
    ) -> float:
        """Get probability of transitioning from one role to another."""
        from_transitions = self.transition_matrix.get(from_role, {})
        return from_transitions.get(to_role, 0.0)

    def get_likely_next_roles(
        self,
        current_role: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get most likely next roles given current role."""
        from_transitions = self.transition_matrix.get(current_role, {})
        sorted_transitions = sorted(
            from_transitions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_transitions[:top_k]


# =============================================================================
# Validation Models
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue found in generated text."""
    type: str                     # LENGTH, VOCABULARY, GRAMMAR, etc.
    message: str                  # Human-readable description
    severity: str                 # HIGH, MEDIUM, LOW
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairAction:
    """An action to repair a validation issue."""
    type: str                     # SUBSTITUTE_WORDS, INSERT_TERMS, etc.

    # Action-specific data
    words: List[str] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    target_range: Tuple[int, int] = (0, 0)
    issues: List[str] = field(default_factory=list)


@dataclass
class SentenceValidationResult:
    """Result of validating a single sentence."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    # Individual check results
    length_ok: bool = True
    complexity_ok: bool = True
    vocabulary_ok: bool = True
    grammar_ok: bool = True
    technical_terms_preserved: bool = True

    # Repair suggestions
    suggested_repairs: List[RepairAction] = field(default_factory=list)

    def get_high_severity_issues(self) -> List[ValidationIssue]:
        """Get only high-severity issues."""
        return [i for i in self.issues if i.severity == "HIGH"]


@dataclass
class ParagraphRepairAction:
    """An action to repair a paragraph-level issue."""
    type: str                     # ADJUST_BURSTINESS, ADJUST_VOCABULARY, etc.

    # Burstiness adjustment
    current: float = 0.0
    target: float = 0.0
    lengths: List[int] = field(default_factory=list)

    # Vocabulary adjustment
    overused: List[Tuple[str, float, float]] = field(default_factory=list)
    underused: List[Tuple[str, float, float]] = field(default_factory=list)

    # Rhythm adjustment
    current_pattern: List[float] = field(default_factory=list)
    target_patterns: List[List[float]] = field(default_factory=list)


@dataclass
class ParagraphValidationResult:
    """Result of validating an entire paragraph."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    # Individual check results
    burstiness_ok: bool = True
    rhythm_pattern_ok: bool = True
    transition_pattern_ok: bool = True
    vocabulary_distribution_ok: bool = True
    structure_ok: bool = True

    # Corpus similarity (0-1)
    corpus_similarity: float = 0.0

    # Repair suggestions
    suggested_repairs: List[ParagraphRepairAction] = field(default_factory=list)

    def get_high_severity_issues(self) -> List[ValidationIssue]:
        """Get only high-severity issues."""
        return [i for i in self.issues if i.severity == "HIGH"]
