"""Hybrid template-guided generation with style constraints.

Combines:
1. Template skeletons as structural scaffolds
2. Style fingerprint for vocabulary guidance
3. Discourse context for coherence
4. LLM for natural language filling

The template provides structural consistency that pure LLM generation
struggles to maintain, especially for longer sentences.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import re

from .models import (
    SentenceTemplate,
    SlotType,
    SentenceType,
    RhetoricalRole,
    LogicalRelation,
)
from .storage import TemplateStore as TemplateStorage
from .fingerprint import StyleFingerprint, StyleConstraints
from .discourse import (
    SentenceContext,
    ParagraphContext,
    DiscourseRelation,
    SentenceRole,
    DISCOURSE_MARKERS,
)
from .filler import Proposition
from ..utils.nlp import split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TemplateMatchCriteria(Enum):
    """How strictly to match templates."""
    EXACT = "exact"       # Match type + role + length
    RELAXED = "relaxed"   # Match type + role
    MINIMAL = "minimal"   # Match type only


@dataclass
class TemplateHint:
    """A template skeleton with guidance for LLM filling."""

    skeleton: str                    # e.g., "[CLAIM], and thus [RESULT]"
    slot_types: List[SlotType]       # What each slot expects
    slot_descriptions: List[str]     # Human-readable descriptions
    target_length: Tuple[int, int]   # (min, max) words
    structural_words: List[str]      # Words that must be preserved
    example_filled: Optional[str]    # Example from corpus

    def to_prompt_section(self) -> str:
        """Convert to prompt format."""
        lines = []
        lines.append("=== STRUCTURAL TEMPLATE ===")
        lines.append(f"Pattern: {self.skeleton}")
        lines.append(f"Target length: {self.target_length[0]}-{self.target_length[1]} words")

        if self.structural_words:
            lines.append(f"Keep these structural words: {', '.join(self.structural_words)}")

        lines.append("")
        lines.append("Slots to fill:")
        for i, (stype, desc) in enumerate(zip(self.slot_types, self.slot_descriptions)):
            lines.append(f"  [{stype.value.upper()}]: {desc}")

        if self.example_filled:
            lines.append("")
            lines.append(f"Example from author: \"{self.example_filled[:100]}...\"")

        return "\n".join(lines)


@dataclass
class HybridPrompt:
    """Complete prompt combining template, style, and context."""

    # Structural scaffold
    template_hint: TemplateHint

    # Content to express
    proposition: Proposition

    # Discourse context
    sentence_role: SentenceRole
    discourse_relation: DiscourseRelation
    previous_sentence: Optional[str]
    paragraph_topic: Optional[str]

    # Style constraints
    preferred_vocabulary: List[str]
    preferred_transitions: List[str]
    characteristic_phrases: List[str]
    framing_patterns: List[str]

    # Author info
    author_name: str

    def to_prompt(self) -> str:
        """Generate the complete LLM prompt."""
        lines = []

        # Header with clear instruction
        lines.append(f"Generate a sentence in the exact style of {self.author_name}.")
        lines.append("Follow the structural template precisely while using the author's vocabulary.")
        lines.append("")

        # Template structure (most important for consistency)
        lines.append(self.template_hint.to_prompt_section())
        lines.append("")

        # Discourse context
        lines.append("=== DISCOURSE CONTEXT ===")
        lines.append(f"Sentence function: {self.sentence_role.value}")

        if self.discourse_relation != DiscourseRelation.NONE:
            lines.append(f"Relation to previous: {self.discourse_relation.value}")
            markers = DISCOURSE_MARKERS.get(self.discourse_relation, [])
            if markers:
                lines.append(f"Appropriate connectors: {', '.join(markers[:4])}")

        if self.previous_sentence:
            lines.append(f"Previous sentence: \"{self.previous_sentence[:80]}...\"")

        if self.paragraph_topic:
            lines.append(f"Paragraph topic: {self.paragraph_topic}")
        lines.append("")

        # Style vocabulary
        lines.append("=== AUTHOR'S VOCABULARY ===")
        if self.preferred_vocabulary:
            lines.append(f"Preferred words: {', '.join(self.preferred_vocabulary[:20])}")
        if self.preferred_transitions:
            lines.append(f"Transition style: {', '.join(self.preferred_transitions[:5])}")
        if self.characteristic_phrases:
            lines.append(f"Characteristic phrases: {'; '.join(self.characteristic_phrases[:5])}")
        if self.framing_patterns:
            lines.append(f"Framing examples: {'; '.join(self.framing_patterns[:3])}")
        lines.append("")

        # The semantic content to express
        lines.append("=== CONTENT TO EXPRESS ===")
        lines.append(f"Subject: {self.proposition.subject}")
        lines.append(f"Predicate: {self.proposition.predicate}")
        if self.proposition.object:
            lines.append(f"Object: {self.proposition.object}")
        lines.append("")

        # Final instruction
        lines.append("=== INSTRUCTIONS ===")
        lines.append("1. Follow the structural template pattern exactly")
        lines.append("2. Fill slots with content that expresses the proposition")
        lines.append("3. Use the author's vocabulary and phrasing style")
        lines.append("4. Maintain the target sentence length")
        lines.append("5. Include appropriate discourse connectors if needed")
        lines.append("")
        lines.append("Generate ONLY the sentence, nothing else:")

        return "\n".join(lines)


class TemplateSelector:
    """Selects best matching template for a generation task."""

    def __init__(self, storage: TemplateStorage):
        self.storage = storage

    def select_template(
        self,
        sentence_type: SentenceType,
        rhetorical_role: RhetoricalRole,
        target_length: int,
        discourse_relation: DiscourseRelation,
        author: Optional[str] = None,
    ) -> Optional[SentenceTemplate]:
        """Select best matching template.

        Args:
            sentence_type: Type of sentence needed.
            rhetorical_role: Role in paragraph.
            target_length: Approximate word count.
            discourse_relation: Relation to previous sentence.
            author: Author to match (if any).

        Returns:
            Best matching template or None.
        """
        # Try exact match first
        templates = self.storage.query_templates(
            author=author,
            sentence_type=sentence_type,
            rhetorical_role=rhetorical_role,
            limit=10,
        )

        if templates:
            # Filter by length similarity
            best = min(
                templates,
                key=lambda t: abs(t.word_count - target_length)
            )
            return best

        # Relax to type + role
        templates = self.storage.query_templates(
            author=author,
            sentence_type=sentence_type,
            limit=10,
        )

        if templates:
            # Prefer templates that match the discourse relation
            relation_matching = [
                t for t in templates
                if self._template_matches_relation(t, discourse_relation)
            ]
            if relation_matching:
                return min(relation_matching, key=lambda t: abs(t.word_count - target_length))
            return min(templates, key=lambda t: abs(t.word_count - target_length))

        # Last resort: any template from author
        templates = self.storage.query_templates(author=author, limit=5)
        if templates:
            return min(templates, key=lambda t: abs(t.word_count - target_length))

        return None

    def _template_matches_relation(
        self,
        template: SentenceTemplate,
        relation: DiscourseRelation,
    ) -> bool:
        """Check if template structure matches discourse relation."""
        skeleton_lower = template.skeleton.lower()

        # Check for relation-appropriate markers
        markers = DISCOURSE_MARKERS.get(relation, [])
        return any(marker in skeleton_lower for marker in markers)


class HybridGenerator:
    """Generates text using template scaffolds with style constraints."""

    def __init__(
        self,
        storage: TemplateStorage,
        fingerprint: StyleFingerprint,
        author_name: str = "the author",
    ):
        """Initialize hybrid generator.

        Args:
            storage: Template storage with author's templates.
            fingerprint: Author's style fingerprint.
            author_name: Name for prompts.
        """
        self.storage = storage
        self.fingerprint = fingerprint
        self.author_name = author_name
        self.selector = TemplateSelector(storage)

    def create_template_hint(
        self,
        template: SentenceTemplate,
    ) -> TemplateHint:
        """Convert template to a hint for the LLM.

        Args:
            template: The selected template.

        Returns:
            TemplateHint with guidance.
        """
        # Extract slot descriptions
        slot_descriptions = []
        for slot in template.slots:
            desc = self._describe_slot(slot.slot_type)
            slot_descriptions.append(desc)

        # Identify structural words to preserve
        structural_words = self._extract_structural_words(template.skeleton)

        # Get target length range
        min_len = max(5, template.word_count - 5)
        max_len = template.word_count + 5

        return TemplateHint(
            skeleton=template.skeleton,
            slot_types=[s.slot_type for s in template.slots],
            slot_descriptions=slot_descriptions,
            target_length=(min_len, max_len),
            structural_words=structural_words,
            example_filled=template.original_text if hasattr(template, 'original_text') else None,
        )

    def build_prompt(
        self,
        proposition: Proposition,
        sentence_context: SentenceContext,
        paragraph_context: ParagraphContext,
        template: Optional[SentenceTemplate] = None,
    ) -> HybridPrompt:
        """Build a hybrid prompt combining template and style.

        Args:
            proposition: What to express.
            sentence_context: Sentence-level context.
            paragraph_context: Paragraph-level context.
            template: Pre-selected template (or will select one).

        Returns:
            HybridPrompt ready for LLM.
        """
        # Select template if not provided
        if template is None:
            # Map sentence role to rhetorical role
            role_map = {
                SentenceRole.TOPIC: RhetoricalRole.CLAIM,
                SentenceRole.CLAIM: RhetoricalRole.CLAIM,
                SentenceRole.SUPPORT: RhetoricalRole.EVIDENCE,
                SentenceRole.EXAMPLE: RhetoricalRole.EVIDENCE,
                SentenceRole.ELABORATION: RhetoricalRole.ELABORATION,
                SentenceRole.TRANSITION: RhetoricalRole.TRANSITION,
                SentenceRole.CONCLUSION: RhetoricalRole.CONCLUSION,
                SentenceRole.CONTRAST: RhetoricalRole.CONTRAST,
                SentenceRole.CONCESSION: RhetoricalRole.CONTRAST,
            }
            rhetorical_role = role_map.get(sentence_context.role, RhetoricalRole.CLAIM)

            # Estimate target length from fingerprint
            target_length = int(self.fingerprint.sentence_length_mean)

            template = self.selector.select_template(
                sentence_type=SentenceType.DECLARATIVE,  # Most common
                rhetorical_role=rhetorical_role,
                target_length=target_length,
                discourse_relation=sentence_context.discourse_relation,
                author=self.author_name,
            )

        # Create template hint
        if template:
            template_hint = self.create_template_hint(template)
        else:
            # Fallback: create a generic hint from style patterns
            template_hint = self._create_fallback_hint(sentence_context)

        # Get previous sentence text
        prev_sent = sentence_context.previous_topic

        # Get paragraph topic
        para_topic = paragraph_context.previous_paragraph_topic

        return HybridPrompt(
            template_hint=template_hint,
            proposition=proposition,
            sentence_role=sentence_context.role,
            discourse_relation=sentence_context.discourse_relation,
            previous_sentence=prev_sent,
            paragraph_topic=para_topic,
            preferred_vocabulary=list(self.fingerprint.distinctive_vocabulary.keys())[:30],
            preferred_transitions=self.fingerprint.preferred_transitions,
            characteristic_phrases=self.fingerprint.characteristic_phrases[:10],
            framing_patterns=self.fingerprint.framing_patterns[:5],
            author_name=self.author_name,
        )

    def _describe_slot(self, slot_type: SlotType) -> str:
        """Get human-readable description of slot type."""
        descriptions = {
            SlotType.SUBJECT: "The subject/agent of the action",
            SlotType.VERB: "The main action or state",
            SlotType.OBJECT: "The object/recipient of the action",
            SlotType.CLAUSE: "A dependent or independent clause",
            SlotType.PREPOSITIONAL: "A prepositional phrase for context",
            SlotType.MODIFIER: "An adjective, adverb, or modifying phrase",
            SlotType.CONNECTOR: "A transition word or phrase",
            SlotType.TEMPORAL: "A time reference or temporal phrase",
            SlotType.CAUSAL: "A cause or reason phrase",
        }
        return descriptions.get(slot_type, "Content to fill")

    def _extract_structural_words(self, skeleton: str) -> List[str]:
        """Extract words that should be preserved from skeleton."""
        # Remove slot placeholders
        text = re.sub(r'\[[A-Z_]+\]', '', skeleton)
        # Get remaining words
        words = text.split()
        # Filter to meaningful structural words
        structural = [
            w.strip('.,;:') for w in words
            if len(w.strip('.,;:')) > 2
        ]
        return structural[:5]  # Top 5 structural words

    def _create_fallback_hint(self, context: SentenceContext) -> TemplateHint:
        """Create a generic template hint when no corpus template matches."""
        # Use style patterns to create a reasonable template
        if context.role == SentenceRole.TOPIC:
            skeleton = "[CLAIM]"
            slot_types = [SlotType.CLAUSE]
            slot_descs = ["The main claim of this paragraph"]
        elif context.role == SentenceRole.EXAMPLE:
            skeleton = "For example, [EXAMPLE]"
            slot_types = [SlotType.CLAUSE]
            slot_descs = ["A specific example or illustration"]
        elif context.role == SentenceRole.CONCLUSION:
            skeleton = "Thus, [CONCLUSION]"
            slot_types = [SlotType.CLAUSE]
            slot_descs = ["The concluding statement"]
        else:
            skeleton = "[STATEMENT]"
            slot_types = [SlotType.CLAUSE]
            slot_descs = ["A supporting statement"]

        return TemplateHint(
            skeleton=skeleton,
            slot_types=slot_types,
            slot_descriptions=slot_descs,
            target_length=(
                int(self.fingerprint.sentence_length_mean - self.fingerprint.sentence_length_std),
                int(self.fingerprint.sentence_length_mean + self.fingerprint.sentence_length_std),
            ),
            structural_words=[],
            example_filled=None,
        )


def create_hybrid_prompt(
    proposition: Proposition,
    template: SentenceTemplate,
    fingerprint: StyleFingerprint,
    sentence_context: SentenceContext,
    paragraph_context: ParagraphContext,
    author_name: str,
) -> str:
    """Convenience function to create a hybrid prompt.

    Args:
        proposition: What to express.
        template: Template to use as scaffold.
        fingerprint: Author's style fingerprint.
        sentence_context: Sentence-level context.
        paragraph_context: Paragraph-level context.
        author_name: Author name.

    Returns:
        Complete prompt string.
    """
    # Create template hint
    slot_descs = []
    for slot in template.slots:
        descs = {
            SlotType.SUBJECT: "Subject/agent",
            SlotType.VERB: "Main action",
            SlotType.OBJECT: "Object/recipient",
            SlotType.CLAUSE: "Dependent clause",
            SlotType.PREPOSITIONAL: "Prepositional phrase",
            SlotType.MODIFIER: "Modifier",
            SlotType.CONNECTOR: "Transition",
            SlotType.TEMPORAL: "Time reference",
            SlotType.CAUSAL: "Cause/reason",
        }
        slot_descs.append(descs.get(slot.slot_type, "Content"))

    hint = TemplateHint(
        skeleton=template.skeleton,
        slot_types=[s.slot_type for s in template.slots],
        slot_descriptions=slot_descs,
        target_length=(max(5, template.word_count - 5), template.word_count + 5),
        structural_words=[],
        example_filled=None,
    )

    prompt = HybridPrompt(
        template_hint=hint,
        proposition=proposition,
        sentence_role=sentence_context.role,
        discourse_relation=sentence_context.discourse_relation,
        previous_sentence=sentence_context.previous_topic,
        paragraph_topic=paragraph_context.previous_paragraph_topic,
        preferred_vocabulary=list(fingerprint.distinctive_vocabulary.keys())[:30],
        preferred_transitions=fingerprint.preferred_transitions,
        characteristic_phrases=fingerprint.characteristic_phrases[:10],
        framing_patterns=fingerprint.framing_patterns[:5],
        author_name=author_name,
    )

    return prompt.to_prompt()
