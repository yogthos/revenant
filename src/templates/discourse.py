"""Discourse-aware context modeling for coherent text generation.

Tracks document structure, paragraph roles, sentence functions, and
provides context for generating coherent, human-like text with proper
backward references and forward setups.

Based on Rhetorical Structure Theory (Mann & Thompson, 1988) and
discourse coherence research.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple
import re

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Enums for Document Structure
# =============================================================================

class DocumentPosition(Enum):
    """Position of paragraph in document."""
    INTRO = "intro"           # Opening paragraph(s)
    BODY = "body"             # Main content
    CONCLUSION = "conclusion" # Closing paragraph(s)


class ParagraphRole(Enum):
    """Rhetorical role of a paragraph."""
    THESIS = "thesis"           # States main claim/argument
    BACKGROUND = "background"   # Provides context/setup
    EVIDENCE = "evidence"       # Supports with facts/examples
    ELABORATION = "elaboration" # Expands on previous point
    TRANSITION = "transition"   # Bridges between topics
    COUNTERPOINT = "counterpoint"  # Addresses objections
    SYNTHESIS = "synthesis"     # Combines multiple points
    CONCLUSION = "conclusion"   # Summarizes/concludes


class SentenceRole(Enum):
    """Rhetorical role of a sentence within paragraph."""
    TOPIC = "topic"           # Topic sentence (main claim of paragraph)
    CLAIM = "claim"           # Makes an assertion
    SUPPORT = "support"       # Provides evidence/reasoning
    EXAMPLE = "example"       # Gives concrete example
    ELABORATION = "elaboration"  # Expands on previous
    CONTRAST = "contrast"     # Introduces opposing view
    CONCESSION = "concession" # Acknowledges limitation
    TRANSITION = "transition" # Links to next idea
    CONCLUSION = "conclusion" # Wraps up paragraph


class DiscourseRelation(Enum):
    """Rhetorical relation to previous unit (RST-based)."""
    NONE = "none"             # First unit, no relation
    ELABORATION = "elaboration"  # Expands on previous
    CONTRAST = "contrast"     # But, however, yet
    CAUSE = "cause"           # Because, since
    RESULT = "result"         # Therefore, thus, consequently
    BACKGROUND = "background" # Provides context
    SEQUENCE = "sequence"     # Then, next, subsequently
    CONCESSION = "concession" # Although, even though
    CONDITION = "condition"   # If, when
    RESTATEMENT = "restatement"  # In other words
    SUMMARY = "summary"       # In conclusion, to summarize
    EXAMPLE = "example"       # For instance, for example


class ReferenceType(Enum):
    """Type of cross-reference in sentence."""
    NONE = "none"
    BACKWARD = "backward"     # References previous content
    FORWARD = "forward"       # Sets up upcoming content
    BOTH = "both"             # Both backward and forward


# =============================================================================
# Discourse Markers by Relation
# =============================================================================

DISCOURSE_MARKERS = {
    DiscourseRelation.ELABORATION: [
        "indeed", "in fact", "moreover", "furthermore", "additionally",
        "what is more", "that is to say", "in other words", "specifically",
    ],
    DiscourseRelation.CONTRAST: [
        "but", "however", "yet", "nevertheless", "nonetheless",
        "on the other hand", "in contrast", "conversely", "whereas",
    ],
    DiscourseRelation.CAUSE: [
        "because", "since", "as", "for", "given that",
        "due to", "owing to", "on account of",
    ],
    DiscourseRelation.RESULT: [
        "therefore", "thus", "hence", "consequently", "accordingly",
        "as a result", "so", "it follows that", "for this reason",
    ],
    DiscourseRelation.BACKGROUND: [
        "historically", "traditionally", "previously",
        "in the past", "before this",
    ],
    DiscourseRelation.SEQUENCE: [
        "first", "second", "third", "then", "next", "subsequently",
        "finally", "lastly", "after this",
    ],
    DiscourseRelation.CONCESSION: [
        "although", "though", "even though", "while", "despite",
        "in spite of", "granted that", "admittedly",
    ],
    DiscourseRelation.CONDITION: [
        "if", "when", "unless", "provided that", "in case",
        "assuming that", "given that",
    ],
    DiscourseRelation.RESTATEMENT: [
        "in other words", "that is", "namely", "to put it another way",
        "to be more precise",
    ],
    DiscourseRelation.SUMMARY: [
        "in conclusion", "to summarize", "in summary", "to conclude",
        "in short", "briefly", "overall", "thus we see",
    ],
    DiscourseRelation.EXAMPLE: [
        "for example", "for instance", "such as", "namely",
        "as an illustration", "to illustrate", "consider",
    ],
}

# Backward reference phrases
BACKWARD_REFERENCE_PHRASES = [
    "as we have seen", "as noted above", "as discussed",
    "this", "these", "such", "the aforementioned",
    "the preceding", "as mentioned", "returning to",
]

# Forward setup phrases
FORWARD_SETUP_PHRASES = [
    "as we shall see", "as will be shown", "below we will",
    "this leads to", "we now turn to", "the following",
    "in the next section", "subsequently",
]


# =============================================================================
# Data Classes for Context Tracking
# =============================================================================

@dataclass
class EntityMention:
    """Tracks an entity mentioned in the text."""
    text: str
    canonical_form: str  # Normalized form
    first_mention_para: int
    first_mention_sent: int
    mention_count: int = 1
    is_key_concept: bool = False


@dataclass
class SentenceContext:
    """Context for a single sentence."""
    position: int              # Position in paragraph (0-indexed)
    total_in_paragraph: int    # Total sentences in paragraph
    role: SentenceRole
    discourse_relation: DiscourseRelation
    reference_type: ReferenceType

    # Content from previous sentence (for coherence)
    previous_topic: Optional[str] = None
    previous_entities: Set[str] = field(default_factory=set)

    # What this sentence should reference or set up
    backward_ref_target: Optional[str] = None  # What to reference from before
    forward_setup_target: Optional[str] = None  # What to set up for later

    # Suggested discourse markers
    suggested_markers: List[str] = field(default_factory=list)

    @property
    def is_first(self) -> bool:
        return self.position == 0

    @property
    def is_last(self) -> bool:
        return self.position == self.total_in_paragraph - 1

    @property
    def is_middle(self) -> bool:
        return not self.is_first and not self.is_last


@dataclass
class ParagraphContext:
    """Context for a paragraph."""
    position: int              # Position in document (0-indexed)
    total_in_document: int     # Total paragraphs
    document_position: DocumentPosition
    role: ParagraphRole
    discourse_relation: DiscourseRelation  # Relation to previous paragraph

    # Content summary from previous paragraph
    previous_paragraph_topic: Optional[str] = None
    previous_paragraph_conclusion: Optional[str] = None
    previous_key_entities: Set[str] = field(default_factory=set)

    # What this paragraph should accomplish
    should_reference_previous: bool = False
    should_setup_next: bool = False
    next_paragraph_topic_hint: Optional[str] = None

    # Sentence contexts within this paragraph
    sentence_contexts: List[SentenceContext] = field(default_factory=list)

    @property
    def is_first(self) -> bool:
        return self.position == 0

    @property
    def is_last(self) -> bool:
        return self.position == self.total_in_document - 1


@dataclass
class DocumentContext:
    """Full document context."""
    title: Optional[str] = None
    thesis: Optional[str] = None
    total_paragraphs: int = 0

    # Track key entities throughout document
    entities: Dict[str, EntityMention] = field(default_factory=dict)

    # Track what's been established
    established_claims: List[str] = field(default_factory=list)
    established_examples: List[str] = field(default_factory=list)

    # Paragraph contexts
    paragraph_contexts: List[ParagraphContext] = field(default_factory=list)


# =============================================================================
# Context Analyzer
# =============================================================================

class DiscourseAnalyzer:
    """Analyzes text structure and extracts discourse context."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def analyze_document(self, paragraphs: List[str]) -> DocumentContext:
        """Analyze full document structure.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            DocumentContext with full structure analysis.
        """
        doc_ctx = DocumentContext(total_paragraphs=len(paragraphs))

        for i, para in enumerate(paragraphs):
            para_ctx = self.analyze_paragraph(
                para,
                position=i,
                total=len(paragraphs),
                previous_context=doc_ctx.paragraph_contexts[-1] if doc_ctx.paragraph_contexts else None,
                doc_context=doc_ctx,
            )
            doc_ctx.paragraph_contexts.append(para_ctx)

            # Extract thesis from first paragraph
            if i == 0 and para_ctx.sentence_contexts:
                # First sentence of first paragraph often contains thesis
                doc_ctx.thesis = self._extract_main_claim(para)

        return doc_ctx

    def analyze_paragraph(
        self,
        paragraph: str,
        position: int,
        total: int,
        previous_context: Optional[ParagraphContext] = None,
        doc_context: Optional[DocumentContext] = None,
    ) -> ParagraphContext:
        """Analyze paragraph structure and role.

        Args:
            paragraph: Paragraph text.
            position: Position in document.
            total: Total paragraphs.
            previous_context: Previous paragraph's context.
            doc_context: Document-level context.

        Returns:
            ParagraphContext with structure analysis.
        """
        # Determine document position
        if position == 0:
            doc_pos = DocumentPosition.INTRO
        elif position == total - 1:
            doc_pos = DocumentPosition.CONCLUSION
        else:
            doc_pos = DocumentPosition.BODY

        # Determine paragraph role
        role = self._classify_paragraph_role(paragraph, doc_pos, position, total)

        # Determine relation to previous
        if previous_context:
            relation = self._classify_paragraph_relation(paragraph, previous_context)
            prev_topic = previous_context.previous_paragraph_topic
            prev_entities = previous_context.previous_key_entities
        else:
            relation = DiscourseRelation.NONE
            prev_topic = None
            prev_entities = set()

        para_ctx = ParagraphContext(
            position=position,
            total_in_document=total,
            document_position=doc_pos,
            role=role,
            discourse_relation=relation,
            previous_paragraph_topic=prev_topic,
            previous_key_entities=prev_entities,
            should_reference_previous=position > 0,
            should_setup_next=position < total - 1,
        )

        # Analyze sentences within paragraph
        sentences = split_into_sentences(paragraph)
        for i, sent in enumerate(sentences):
            sent_ctx = self.analyze_sentence(
                sent,
                position=i,
                total=len(sentences),
                paragraph_context=para_ctx,
                previous_sentence=sentences[i-1] if i > 0 else None,
            )
            para_ctx.sentence_contexts.append(sent_ctx)

        # Extract paragraph topic for next paragraph's reference
        if sentences:
            para_ctx.previous_paragraph_topic = self._extract_main_claim(sentences[0])
            if len(sentences) > 1:
                para_ctx.previous_paragraph_conclusion = sentences[-1]

        # Extract key entities
        para_ctx.previous_key_entities = self._extract_key_entities(paragraph)

        return para_ctx

    def analyze_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
        paragraph_context: ParagraphContext,
        previous_sentence: Optional[str] = None,
    ) -> SentenceContext:
        """Analyze sentence role and relations.

        Args:
            sentence: Sentence text.
            position: Position in paragraph.
            total: Total sentences in paragraph.
            paragraph_context: Parent paragraph context.
            previous_sentence: Previous sentence text.

        Returns:
            SentenceContext with role and relation analysis.
        """
        # Determine sentence role based on position and content
        role = self._classify_sentence_role(
            sentence, position, total, paragraph_context
        )

        # Determine discourse relation to previous
        if previous_sentence:
            relation = self._classify_sentence_relation(sentence, previous_sentence)
            prev_topic = self._extract_main_claim(previous_sentence)
            prev_entities = self._extract_key_entities(previous_sentence)
        else:
            relation = DiscourseRelation.NONE
            prev_topic = None
            prev_entities = set()

        # Determine reference type
        ref_type = self._detect_reference_type(sentence)

        # Get suggested discourse markers
        suggested = DISCOURSE_MARKERS.get(relation, [])[:3]

        return SentenceContext(
            position=position,
            total_in_paragraph=total,
            role=role,
            discourse_relation=relation,
            reference_type=ref_type,
            previous_topic=prev_topic,
            previous_entities=prev_entities,
            suggested_markers=suggested,
        )

    def _classify_paragraph_role(
        self,
        paragraph: str,
        doc_position: DocumentPosition,
        position: int,
        total: int,
    ) -> ParagraphRole:
        """Classify the rhetorical role of a paragraph."""
        para_lower = paragraph.lower()

        # Position-based heuristics
        if doc_position == DocumentPosition.INTRO:
            if position == 0:
                return ParagraphRole.THESIS
            return ParagraphRole.BACKGROUND

        if doc_position == DocumentPosition.CONCLUSION:
            return ParagraphRole.CONCLUSION

        # Content-based heuristics
        if any(marker in para_lower for marker in ["for example", "for instance", "such as"]):
            return ParagraphRole.EVIDENCE

        if any(marker in para_lower for marker in ["however", "but", "on the other hand"]):
            return ParagraphRole.COUNTERPOINT

        if any(marker in para_lower for marker in ["therefore", "thus", "consequently"]):
            return ParagraphRole.SYNTHESIS

        if any(marker in para_lower for marker in ["furthermore", "moreover", "additionally"]):
            return ParagraphRole.ELABORATION

        return ParagraphRole.ELABORATION  # Default

    def _classify_paragraph_relation(
        self,
        paragraph: str,
        previous_context: ParagraphContext,
    ) -> DiscourseRelation:
        """Classify relation between paragraphs."""
        para_lower = paragraph.lower()
        first_sentence = split_into_sentences(paragraph)[0].lower() if paragraph else ""

        # Check for explicit markers at paragraph start
        for relation, markers in DISCOURSE_MARKERS.items():
            for marker in markers:
                if first_sentence.startswith(marker) or f" {marker}" in first_sentence[:50]:
                    return relation

        # Default based on position
        if previous_context.role == ParagraphRole.THESIS:
            return DiscourseRelation.ELABORATION

        return DiscourseRelation.SEQUENCE

    def _classify_sentence_role(
        self,
        sentence: str,
        position: int,
        total: int,
        paragraph_context: ParagraphContext,
    ) -> SentenceRole:
        """Classify the role of a sentence within its paragraph."""
        sent_lower = sentence.lower()

        # Position-based
        if position == 0:
            return SentenceRole.TOPIC

        if position == total - 1:
            if paragraph_context.document_position == DocumentPosition.CONCLUSION:
                return SentenceRole.CONCLUSION
            if any(m in sent_lower for m in ["therefore", "thus", "consequently"]):
                return SentenceRole.CONCLUSION
            return SentenceRole.CONCLUSION

        # Content-based
        if any(m in sent_lower for m in ["for example", "for instance"]):
            return SentenceRole.EXAMPLE

        if any(m in sent_lower for m in ["however", "but", "yet"]):
            return SentenceRole.CONTRAST

        if any(m in sent_lower for m in ["although", "while", "despite"]):
            return SentenceRole.CONCESSION

        if any(m in sent_lower for m in ["furthermore", "moreover"]):
            return SentenceRole.ELABORATION

        return SentenceRole.SUPPORT  # Default for middle sentences

    def _classify_sentence_relation(
        self,
        sentence: str,
        previous: str,
    ) -> DiscourseRelation:
        """Classify discourse relation between sentences."""
        sent_lower = sentence.lower()

        # Check for explicit markers
        for relation, markers in DISCOURSE_MARKERS.items():
            for marker in markers:
                if sent_lower.startswith(marker):
                    return relation
                if sent_lower.startswith(marker + ","):
                    return relation

        # Check if continues same topic (elaboration)
        prev_entities = self._extract_key_entities(previous)
        sent_entities = self._extract_key_entities(sentence)

        if prev_entities & sent_entities:
            return DiscourseRelation.ELABORATION

        return DiscourseRelation.SEQUENCE  # Default

    def _detect_reference_type(self, sentence: str) -> ReferenceType:
        """Detect if sentence references previous or sets up future content."""
        sent_lower = sentence.lower()

        has_backward = any(phrase in sent_lower for phrase in BACKWARD_REFERENCE_PHRASES)
        has_forward = any(phrase in sent_lower for phrase in FORWARD_SETUP_PHRASES)

        if has_backward and has_forward:
            return ReferenceType.BOTH
        if has_backward:
            return ReferenceType.BACKWARD
        if has_forward:
            return ReferenceType.FORWARD
        return ReferenceType.NONE

    def _extract_main_claim(self, text: str) -> str:
        """Extract the main claim/topic from text."""
        doc = self.nlp(text)

        # Find the root verb and its subject
        for token in doc:
            if token.dep_ == "ROOT":
                # Get subject
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                if subjects:
                    subj = subjects[0]
                    subj_text = " ".join([t.text for t in subj.subtree])
                    return f"{subj_text} {token.text}"

        # Fallback: return first noun phrase
        for chunk in doc.noun_chunks:
            return chunk.text

        return text[:50]  # Last resort

    def _extract_key_entities(self, text: str) -> Set[str]:
        """Extract key entities from text."""
        doc = self.nlp(text)
        entities = set()

        # Named entities
        for ent in doc.ents:
            entities.add(ent.text.lower())

        # Key noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # Not too long
                entities.add(chunk.root.lemma_.lower())

        return entities


# =============================================================================
# Context-Aware Prompt Builder
# =============================================================================

class DiscoursePromptBuilder:
    """Builds context-aware prompts for LLM generation."""

    def build_sentence_prompt(
        self,
        proposition: str,
        sentence_context: SentenceContext,
        paragraph_context: ParagraphContext,
        style_constraints: Optional[Dict] = None,
    ) -> str:
        """Build a context-aware prompt for sentence generation.

        Args:
            proposition: The semantic content to express.
            sentence_context: Context for this sentence.
            paragraph_context: Context for parent paragraph.
            style_constraints: Style fingerprint constraints.

        Returns:
            Prompt string for LLM.
        """
        parts = []

        # Document position context
        parts.append(f"=== DOCUMENT CONTEXT ===")
        parts.append(f"Position: {paragraph_context.document_position.value} paragraph")
        parts.append(f"Paragraph {paragraph_context.position + 1} of {paragraph_context.total_in_document}")

        # Paragraph role
        parts.append(f"\n=== PARAGRAPH ROLE ===")
        parts.append(f"Role: {paragraph_context.role.value}")
        if paragraph_context.discourse_relation != DiscourseRelation.NONE:
            parts.append(f"Relation to previous paragraph: {paragraph_context.discourse_relation.value}")

        # Previous paragraph context
        if paragraph_context.previous_paragraph_topic:
            parts.append(f"Previous paragraph topic: {paragraph_context.previous_paragraph_topic}")
        if paragraph_context.previous_key_entities:
            parts.append(f"Previously introduced concepts: {', '.join(list(paragraph_context.previous_key_entities)[:5])}")

        # Sentence position and role
        parts.append(f"\n=== SENTENCE CONTEXT ===")
        parts.append(f"Sentence {sentence_context.position + 1} of {sentence_context.total_in_paragraph}")
        parts.append(f"Role: {sentence_context.role.value}")

        if sentence_context.discourse_relation != DiscourseRelation.NONE:
            parts.append(f"Relation to previous: {sentence_context.discourse_relation.value}")
            if sentence_context.suggested_markers:
                parts.append(f"Suggested connectors: {', '.join(sentence_context.suggested_markers)}")

        # Reference guidance
        if sentence_context.reference_type == ReferenceType.BACKWARD:
            parts.append(f"Should reference: {sentence_context.previous_topic or 'previous point'}")
        elif sentence_context.reference_type == ReferenceType.FORWARD:
            parts.append(f"Should set up: upcoming discussion")
        elif sentence_context.reference_type == ReferenceType.BOTH:
            parts.append(f"Should reference previous and set up next")

        # Previous sentence context
        if sentence_context.previous_topic:
            parts.append(f"Previous sentence topic: {sentence_context.previous_topic}")
        if sentence_context.previous_entities:
            parts.append(f"Active entities: {', '.join(list(sentence_context.previous_entities)[:3])}")

        # Style constraints
        if style_constraints:
            parts.append(f"\n=== STYLE CONSTRAINTS ===")
            if "sentence_length" in style_constraints:
                parts.append(f"Target length: {style_constraints['sentence_length']} words")
            if "vocabulary" in style_constraints:
                parts.append(f"Preferred vocabulary: {', '.join(style_constraints['vocabulary'][:10])}")
            if "transitions" in style_constraints:
                parts.append(f"Preferred transitions: {', '.join(style_constraints['transitions'][:5])}")
            if "framing" in style_constraints:
                parts.append(f"Framing style: {style_constraints['framing']}")

        # The actual task
        parts.append(f"\n=== TASK ===")
        parts.append(f"Express this proposition: {proposition}")

        # Specific guidance based on role
        role_guidance = self._get_role_guidance(sentence_context, paragraph_context)
        if role_guidance:
            parts.append(f"\nGuidance: {role_guidance}")

        return "\n".join(parts)

    def _get_role_guidance(
        self,
        sentence_context: SentenceContext,
        paragraph_context: ParagraphContext,
    ) -> str:
        """Get specific guidance based on sentence/paragraph role."""
        guidance = []

        # Paragraph-level guidance
        if paragraph_context.document_position == DocumentPosition.INTRO:
            if paragraph_context.position == 0:
                guidance.append("This opens the document - establish the main topic clearly")
        elif paragraph_context.document_position == DocumentPosition.CONCLUSION:
            guidance.append("This is a concluding paragraph - synthesize and wrap up")

        # Sentence-level guidance
        if sentence_context.role == SentenceRole.TOPIC:
            guidance.append("This is the topic sentence - state the paragraph's main point")
        elif sentence_context.role == SentenceRole.EXAMPLE:
            guidance.append("Provide a concrete example or illustration")
        elif sentence_context.role == SentenceRole.CONCLUSION:
            guidance.append("Conclude this paragraph's argument")
        elif sentence_context.role == SentenceRole.TRANSITION:
            guidance.append("Bridge to the next idea")

        # Discourse relation guidance
        if sentence_context.discourse_relation == DiscourseRelation.CONTRAST:
            guidance.append("Introduce a contrasting point")
        elif sentence_context.discourse_relation == DiscourseRelation.RESULT:
            guidance.append("Draw a conclusion from the preceding point")
        elif sentence_context.discourse_relation == DiscourseRelation.ELABORATION:
            guidance.append("Expand on the previous point")

        return " | ".join(guidance) if guidance else ""

    def build_paragraph_plan(
        self,
        propositions: List[str],
        paragraph_context: ParagraphContext,
    ) -> List[Dict]:
        """Build a plan for generating a paragraph.

        Args:
            propositions: List of propositions to express.
            paragraph_context: Context for this paragraph.

        Returns:
            List of sentence plans with roles and relations.
        """
        plans = []
        n = len(propositions)

        for i, prop in enumerate(propositions):
            # Determine sentence role
            if i == 0:
                role = SentenceRole.TOPIC
                relation = DiscourseRelation.NONE
            elif i == n - 1:
                role = SentenceRole.CONCLUSION
                relation = DiscourseRelation.RESULT
            else:
                role = SentenceRole.SUPPORT
                relation = DiscourseRelation.ELABORATION

            # Determine if should reference or set up
            ref_type = ReferenceType.NONE
            if i > 0 and i < n - 1:
                ref_type = ReferenceType.BACKWARD  # Middle sentences should connect
            elif i == n - 1:
                ref_type = ReferenceType.BACKWARD  # Conclusion references what came before

            plans.append({
                "position": i,
                "proposition": prop,
                "role": role,
                "relation": relation,
                "reference_type": ref_type,
                "suggested_markers": DISCOURSE_MARKERS.get(relation, [])[:3],
            })

        return plans
