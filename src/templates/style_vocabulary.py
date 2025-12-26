"""Style vocabulary classification using spaCy NLP.

Separates transferable style elements from content-specific terms
using linguistic analysis rather than hardcoded word lists.

Transferable (apply to any content):
- Function words (detected by POS: DET, ADP, CCONJ, SCONJ, AUX)
- Discourse connectors (detected by dependency: mark, cc, advmod at sentence start)
- General abstract vocabulary (detected by word vectors + frequency)
- Sentence structure patterns

NOT transferable (topic-specific):
- Named entities (detected by NER)
- Domain-specific nouns (detected by low generality score)
- Proper nouns and specific references
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
from enum import Enum
import re

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VocabularyCategory(Enum):
    """Categories for vocabulary classification."""
    FUNCTION = "function"           # Always transferable (the, and, but, if)
    TRANSITION = "transition"       # Always transferable (thus, therefore, however)
    STRUCTURE = "structure"         # Sentence starters, framing patterns
    GENERAL_VERB = "general_verb"   # Common verbs (is, becomes, requires)
    GENERAL_ADJ = "general_adj"     # Common adjectives (important, necessary)
    GENERAL_NOUN = "general_noun"   # Abstract nouns (process, development)
    TOPIC_SPECIFIC = "topic_specific"  # Domain terms detected by NLP
    NAMED_ENTITY = "named_entity"   # Proper nouns detected by NER


@dataclass
class ClassifiedVocabulary:
    """Vocabulary classified by transferability using NLP."""

    # Always transfer (pure style)
    transitions: Dict[str, float] = field(default_factory=dict)
    function_words: Dict[str, float] = field(default_factory=dict)

    # Transfer when contextually appropriate
    general_verbs: Dict[str, float] = field(default_factory=dict)
    general_adjectives: Dict[str, float] = field(default_factory=dict)
    general_nouns: Dict[str, float] = field(default_factory=dict)

    # DO NOT transfer (content-specific)
    topic_specific: Set[str] = field(default_factory=set)
    named_entities: Set[str] = field(default_factory=set)

    # Structural patterns (abstracted)
    structural_patterns: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class AbstractTemplate:
    """A template with content-specific terms abstracted out."""

    abstract_skeleton: str
    placeholder_types: Dict[str, str]
    structural_words: List[str]
    original_example: Optional[str] = None
    pattern_type: Optional[str] = None

    def to_prompt_section(self) -> str:
        """Convert to prompt format."""
        lines = []
        lines.append("=== SENTENCE STRUCTURE ===")
        lines.append(f"Pattern: {self.abstract_skeleton}")

        if self.structural_words:
            lines.append(f"Structural words to use: {', '.join(self.structural_words)}")

        lines.append("")
        lines.append("Fill these slots with YOUR content:")
        for placeholder, description in self.placeholder_types.items():
            lines.append(f"  {placeholder}: {description}")

        return "\n".join(lines)


class VocabularyClassifier:
    """Classifies vocabulary using spaCy NLP analysis."""

    def __init__(self):
        self._nlp = None
        # POS tags that indicate function/structural words
        self.function_pos = {"DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART", "PRON"}
        # POS tags for content words
        self.content_pos = {"NOUN", "VERB", "ADJ", "ADV"}

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def classify_corpus(self, paragraphs: List[str]) -> ClassifiedVocabulary:
        """Classify vocabulary from corpus using NLP.

        Args:
            paragraphs: Author's corpus paragraphs.

        Returns:
            ClassifiedVocabulary with NLP-based classification.
        """
        result = ClassifiedVocabulary()

        word_counts = Counter()
        pos_counts = {"VERB": Counter(), "ADJ": Counter(), "NOUN": Counter(), "ADV": Counter()}
        transition_candidates = Counter()
        function_word_counts = Counter()
        entity_texts = set()
        pattern_counts = Counter()

        for para in paragraphs:
            doc = self.nlp(para)

            # Collect named entities (never transfer)
            for ent in doc.ents:
                entity_texts.add(ent.text.lower())
                # Also add individual tokens from multi-word entities
                for token in ent:
                    entity_texts.add(token.text.lower())

            # Process sentences for patterns and transitions
            for sent in doc.sents:
                sent_tokens = list(sent)
                if not sent_tokens:
                    continue

                # Detect sentence-initial transitions using dependency parsing
                first_tokens = sent_tokens[:3]
                for token in first_tokens:
                    if self._is_discourse_connector(token):
                        transition_candidates[token.text.lower()] += 1

                # Detect structural patterns
                pattern = self._extract_sentence_pattern(sent)
                if pattern:
                    pattern_counts[pattern] += 1

            # Classify each token
            for token in doc:
                if token.is_space or token.is_punct:
                    continue

                word = token.text.lower()
                lemma = token.lemma_.lower()
                word_counts[word] += 1

                # Function words by POS
                if token.pos_ in self.function_pos:
                    function_word_counts[word] += 1

                # Content words by POS
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_][lemma] += 1

                # Detect transitions by dependency and position
                if self._is_discourse_connector(token):
                    transition_candidates[word] += 1

        total_words = sum(word_counts.values())
        if total_words == 0:
            return result

        # Build function word frequencies
        for word, count in function_word_counts.items():
            result.function_words[word] = count / total_words

        # Build transition frequencies (only confirmed connectors)
        for word, count in transition_candidates.items():
            if count >= 2:  # Appears multiple times as connector
                result.transitions[word] = count / total_words

        # Classify content vocabulary
        for pos, counts in pos_counts.items():
            for word, count in counts.most_common(200):
                # Skip if it's a named entity
                if word in entity_texts:
                    result.named_entities.add(word)
                    continue

                freq = count / total_words

                # Determine if word is general or topic-specific
                is_general = self._is_general_word(word, pos)

                if pos == "VERB":
                    if is_general:
                        result.general_verbs[word] = freq
                    else:
                        result.topic_specific.add(word)
                elif pos == "ADJ":
                    if is_general:
                        result.general_adjectives[word] = freq
                    else:
                        result.topic_specific.add(word)
                elif pos == "NOUN":
                    if is_general:
                        result.general_nouns[word] = freq
                    else:
                        result.topic_specific.add(word)

        # Store named entities
        result.named_entities = entity_texts

        # Store structural patterns
        result.structural_patterns = [
            (pattern, count) for pattern, count in pattern_counts.most_common(20)
        ]

        return result

    def _is_discourse_connector(self, token) -> bool:
        """Detect if token is a discourse connector using spaCy features."""
        # Check dependency relations that indicate discourse function
        discourse_deps = {"cc", "mark", "advmod", "prep"}

        if token.dep_ in discourse_deps:
            # Additional check: should be near sentence start or after punctuation
            if token.i == 0 or (token.i > 0 and token.nbor(-1).is_punct):
                return True

        # Check for adverbs that commonly function as connectors
        if token.pos_ == "ADV" and token.dep_ == "advmod":
            # Check if it modifies the root or is sentence-initial
            if token.head.dep_ == "ROOT" or token.i < 3:
                return True

        # Subordinating conjunctions are always connectors
        if token.pos_ == "SCONJ":
            return True

        # Coordinating conjunctions at clause boundaries
        if token.pos_ == "CCONJ" and token.dep_ == "cc":
            return True

        return False

    def _is_general_word(self, word: str, pos: str) -> bool:
        """Determine if word is general/abstract using spaCy word vectors.

        General words have:
        - High frequency across diverse contexts
        - Abstract/general meaning (measured by vector similarity to abstract concepts)
        - Not domain-specific
        """
        # Get word vector if available
        doc = self.nlp(word)
        if not doc or not doc[0].has_vector:
            # Fallback: use morphological heuristics
            return self._is_general_by_morphology(word, pos)

        token = doc[0]

        # Check vector similarity to general concept words
        # Words similar to abstract concepts are more transferable
        abstract_anchors = ["thing", "process", "state", "action", "quality"]
        concrete_anchors = ["person", "place", "organization", "event"]

        abstract_sim = 0
        concrete_sim = 0

        for anchor in abstract_anchors:
            anchor_doc = self.nlp(anchor)
            if anchor_doc and anchor_doc[0].has_vector:
                abstract_sim += token.similarity(anchor_doc[0])

        for anchor in concrete_anchors:
            anchor_doc = self.nlp(anchor)
            if anchor_doc and anchor_doc[0].has_vector:
                concrete_sim += token.similarity(anchor_doc[0])

        # Normalize
        abstract_sim /= len(abstract_anchors)
        concrete_sim /= len(concrete_anchors)

        # Word is general if more similar to abstract concepts
        return abstract_sim > concrete_sim

    def _is_general_by_morphology(self, word: str, pos: str) -> bool:
        """Fallback: determine generality by morphological features."""
        doc = self.nlp(word)
        if not doc:
            return False

        token = doc[0]

        # Proper nouns are not general
        if token.pos_ == "PROPN":
            return False

        # Very short words tend to be general
        if len(word) <= 4:
            return True

        # Words with certain suffixes tend to be abstract/general
        abstract_suffixes = ("tion", "ment", "ness", "ity", "ism", "ance", "ence")
        if word.endswith(abstract_suffixes):
            return True

        # For verbs, common auxiliary-like verbs are general
        if pos == "VERB":
            common_verbs = {"be", "have", "do", "make", "take", "give", "get",
                           "come", "go", "know", "think", "see", "want", "use",
                           "find", "tell", "become", "leave", "put", "mean",
                           "keep", "let", "begin", "seem", "help", "show",
                           "hear", "play", "run", "move", "hold", "bring",
                           "happen", "write", "provide", "stand", "lose",
                           "meet", "include", "continue", "set", "learn",
                           "change", "lead", "understand", "develop", "turn",
                           "start", "need", "feel", "work", "call", "try"}
            return word in common_verbs or token.lemma_ in common_verbs

        return True  # Default to general for unknown words

    def _extract_sentence_pattern(self, sent) -> Optional[str]:
        """Extract abstract sentence pattern using dependency parsing."""
        tokens = list(sent)
        if not tokens:
            return None

        # Find the root
        root = None
        for token in tokens:
            if token.dep_ == "ROOT":
                root = token
                break

        if not root:
            return None

        # Build pattern based on dependency structure
        pattern_parts = []

        # Check for sentence-initial elements
        first_token = tokens[0]
        if first_token.pos_ == "SCONJ":
            pattern_parts.append("CONDITIONAL")
        elif first_token.pos_ == "ADV" and self._is_discourse_connector(first_token):
            pattern_parts.append("CONNECTOR")
        elif first_token.text.lower() in ("it", "there"):
            pattern_parts.append("EXPLETIVE")

        # Analyze root structure
        has_subject = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
        has_object = any(c.dep_ in ("dobj", "attr", "ccomp") for c in root.children)
        has_complement = any(c.dep_ in ("xcomp", "acomp", "oprd") for c in root.children)

        if has_subject:
            pattern_parts.append("SUBJ")
        pattern_parts.append("VERB")
        if has_object:
            pattern_parts.append("OBJ")
        if has_complement:
            pattern_parts.append("COMP")

        return "_".join(pattern_parts) if pattern_parts else None


class TemplateAbstractor:
    """Abstracts content-specific terms from templates using NLP."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def abstract_template(
        self,
        skeleton: str,
        original_text: str,
    ) -> AbstractTemplate:
        """Abstract content-specific terms using NER and POS analysis.

        Args:
            skeleton: Template skeleton with slots.
            original_text: Original sentence for reference.

        Returns:
            AbstractTemplate with content-specific terms replaced.
        """
        abstract_skeleton = skeleton
        placeholder_types = {}
        structural_words = []

        doc = self.nlp(original_text)

        # Find named entities to abstract
        for ent in doc.ents:
            placeholder = f"[{ent.label_}]"
            # Replace in skeleton if present
            if ent.text in skeleton or ent.text.lower() in skeleton.lower():
                abstract_skeleton = re.sub(
                    r'\b' + re.escape(ent.text) + r'\b',
                    placeholder,
                    abstract_skeleton,
                    flags=re.IGNORECASE
                )
                placeholder_types[placeholder] = self._describe_entity(ent.label_)

        # Find proper nouns not caught by NER
        for token in doc:
            if token.pos_ == "PROPN" and token.text not in str(placeholder_types):
                placeholder = "[ENTITY]"
                if token.text in skeleton:
                    abstract_skeleton = re.sub(
                        r'\b' + re.escape(token.text) + r'\b',
                        placeholder,
                        abstract_skeleton
                    )
                    placeholder_types[placeholder] = "Entity or proper noun from your source"

        # Extract structural words (function words that should be preserved)
        skel_doc = self.nlp(skeleton)
        for token in skel_doc:
            # Skip placeholders
            if token.text.startswith('[') and token.text.endswith(']'):
                continue

            # Keep function words and discourse connectors
            if token.pos_ in ("ADP", "CCONJ", "SCONJ", "ADV", "DET", "AUX", "PART"):
                if len(token.text) > 1 and not token.is_punct:
                    structural_words.append(token.text.lower())

        # Detect pattern type from dependency structure
        pattern_type = self._detect_pattern_type(doc)

        return AbstractTemplate(
            abstract_skeleton=abstract_skeleton,
            placeholder_types=placeholder_types,
            structural_words=list(set(structural_words)),
            original_example=original_text,
            pattern_type=pattern_type,
        )

    def _describe_entity(self, label: str) -> str:
        """Get description for entity type."""
        descriptions = {
            "PERSON": "Person or agent from your source",
            "ORG": "Organization or group from your source",
            "GPE": "Location or geopolitical entity from your source",
            "NORP": "Nationality, group, or ideology from your source",
            "EVENT": "Event from your source",
            "WORK_OF_ART": "Work or creation from your source",
            "LAW": "Law or regulation from your source",
            "DATE": "Date or time from your source",
            "PRODUCT": "Product or object from your source",
        }
        return descriptions.get(label, f"{label} from your source")

    def _detect_pattern_type(self, doc) -> Optional[str]:
        """Detect sentence pattern type from dependency parse."""
        # Find root
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if not root:
            return None

        # Check for common patterns
        first_token = list(doc)[0] if doc else None

        if first_token:
            # Expletive constructions
            if first_token.text.lower() == "it" and root.lemma_ == "be":
                return "EXPLETIVE_IT"
            if first_token.text.lower() == "there" and root.lemma_ in ("be", "exist"):
                return "EXISTENTIAL"

            # Conditional
            if first_token.pos_ == "SCONJ":
                return "CONDITIONAL"

        # Check for reporting/claiming verbs
        claiming_verbs = {"hold", "believe", "argue", "maintain", "contend",
                         "assert", "claim", "state", "say", "think"}
        if root.lemma_ in claiming_verbs:
            # Check if subject is an agent
            for child in root.children:
                if child.dep_ == "nsubj":
                    return "AUTHORITY_CLAIM"

        # Check for necessity/importance patterns
        if root.lemma_ == "be":
            for child in root.children:
                if child.dep_ == "acomp" and child.lemma_ in ("necessary", "important",
                                                               "essential", "clear", "evident"):
                    return "NECESSITY"

        return "DECLARATIVE"  # Default


@dataclass
class TransferableStyle:
    """Style elements that can be transferred to any content."""

    sentence_length_mean: float
    sentence_length_std: float
    burstiness: float
    transitions: Dict[str, float]
    general_verbs: Dict[str, float]
    general_adjectives: Dict[str, float]
    general_nouns: Dict[str, float]
    patterns: List[AbstractTemplate]
    punctuation_freq: Dict[str, float]

    def get_vocabulary_for_prompt(self, max_words: int = 30) -> Dict[str, List[str]]:
        """Get vocabulary lists for prompting."""
        return {
            "transitions": sorted(self.transitions.keys(),
                                  key=lambda x: self.transitions[x],
                                  reverse=True)[:10],
            "verbs": sorted(self.general_verbs.keys(),
                           key=lambda x: self.general_verbs[x],
                           reverse=True)[:max_words // 3],
            "adjectives": sorted(self.general_adjectives.keys(),
                                key=lambda x: self.general_adjectives[x],
                                reverse=True)[:max_words // 3],
            "nouns": sorted(self.general_nouns.keys(),
                           key=lambda x: self.general_nouns[x],
                           reverse=True)[:max_words // 3],
        }

    def to_prompt_section(self) -> str:
        """Convert to prompt format."""
        vocab = self.get_vocabulary_for_prompt()

        lines = []
        lines.append("=== STYLE CONSTRAINTS ===")
        lines.append(f"Target sentence length: {int(self.sentence_length_mean)} words (±{int(self.sentence_length_std)})")
        lines.append("")

        if vocab["transitions"]:
            lines.append("Transition words: " + ", ".join(vocab["transitions"]))

        lines.append("")
        lines.append("General vocabulary preferences:")
        if vocab["verbs"]:
            lines.append(f"  Verbs: {', '.join(vocab['verbs'])}")
        if vocab["adjectives"]:
            lines.append(f"  Adjectives: {', '.join(vocab['adjectives'])}")
        if vocab["nouns"]:
            lines.append(f"  Abstract nouns: {', '.join(vocab['nouns'])}")

        return "\n".join(lines)
