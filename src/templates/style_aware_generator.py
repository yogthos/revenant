"""Style-aware text generation with discourse context.

Combines:
1. Discourse context (document/paragraph/sentence structure)
2. Style fingerprint (statistical author voice)
3. LLM prompting with constraints
4. Verification and repair loop

This is the main orchestration module for generating human-like,
author-specific text.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

from .discourse import (
    DiscourseAnalyzer,
    DiscoursePromptBuilder,
    DocumentContext,
    ParagraphContext,
    SentenceContext,
    DocumentPosition,
    ParagraphRole,
    SentenceRole,
    DiscourseRelation,
    ReferenceType,
    DISCOURSE_MARKERS,
)
from .fingerprint import (
    StyleFingerprintExtractor,
    StyleFingerprint,
    StyleConstraints,
    StyleVerifier,
    StyleVerification,
)
from .filler import Proposition
from ..utils.nlp import split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationContext:
    """Full context for generating a single sentence."""

    # What to say
    proposition: Proposition

    # Where we are in the document
    document_position: DocumentPosition
    paragraph_index: int
    paragraph_total: int
    sentence_index: int
    sentence_total: int

    # Structural roles
    paragraph_role: ParagraphRole
    sentence_role: SentenceRole
    discourse_relation: DiscourseRelation
    reference_type: ReferenceType

    # Previous context for coherence
    previous_sentence: Optional[str] = None
    previous_paragraph_topic: Optional[str] = None
    document_thesis: Optional[str] = None

    # Entities to track
    active_entities: List[str] = field(default_factory=list)
    introduced_entities: List[str] = field(default_factory=list)

    # Style constraints
    style_constraints: Optional[StyleConstraints] = None

    def to_prompt(self, author_name: str = "the author") -> str:
        """Convert context to LLM prompt."""
        lines = []

        # Header
        lines.append(f"Generate a sentence in the style of {author_name}.")
        lines.append("")

        # Document structure
        lines.append("=== DOCUMENT STRUCTURE ===")
        lines.append(f"This is paragraph {self.paragraph_index + 1} of {self.paragraph_total}")
        lines.append(f"Document section: {self.document_position.value.upper()}")
        lines.append(f"Paragraph role: {self.paragraph_role.value}")
        lines.append("")

        # Sentence position
        lines.append("=== SENTENCE POSITION ===")
        lines.append(f"Sentence {self.sentence_index + 1} of {self.sentence_total} in this paragraph")
        lines.append(f"Sentence function: {self.sentence_role.value}")

        if self.discourse_relation != DiscourseRelation.NONE:
            lines.append(f"Relation to previous: {self.discourse_relation.value}")
            markers = DISCOURSE_MARKERS.get(self.discourse_relation, [])
            if markers:
                lines.append(f"Consider using: {', '.join(markers[:4])}")
        lines.append("")

        # Coherence requirements
        lines.append("=== COHERENCE REQUIREMENTS ===")
        if self.reference_type == ReferenceType.BACKWARD:
            lines.append("- Should reference or connect to the previous point")
            if self.previous_sentence:
                lines.append(f"- Previous sentence: \"{self.previous_sentence[:80]}...\"")
        elif self.reference_type == ReferenceType.FORWARD:
            lines.append("- Should set up or foreshadow what comes next")
        elif self.reference_type == ReferenceType.BOTH:
            lines.append("- Should both reference previous and set up next")

        if self.active_entities:
            lines.append(f"- Active topics/entities: {', '.join(self.active_entities[:5])}")

        if self.previous_paragraph_topic and self.sentence_index == 0:
            lines.append(f"- Previous paragraph discussed: {self.previous_paragraph_topic}")
        lines.append("")

        # Style constraints
        if self.style_constraints:
            lines.append("=== STYLE CONSTRAINTS ===")
            lines.append(self.style_constraints.to_prompt_string())
            lines.append("")

        # The task
        lines.append("=== PROPOSITION TO EXPRESS ===")
        prop = self.proposition
        lines.append(f"Subject: {prop.subject}")
        lines.append(f"Predicate: {prop.predicate}")
        if prop.object:
            lines.append(f"Object: {prop.object}")
        if hasattr(prop, 'modifiers') and prop.modifiers:
            lines.append(f"Modifiers: {', '.join(prop.modifiers)}")
        lines.append("")

        # Role-specific guidance
        guidance = self._get_role_guidance()
        if guidance:
            lines.append("=== GUIDANCE ===")
            lines.append(guidance)
            lines.append("")

        lines.append("Generate ONLY the sentence, nothing else:")

        return "\n".join(lines)

    def _get_role_guidance(self) -> str:
        """Get specific guidance based on position and role."""
        parts = []

        # Document position guidance
        if self.document_position == DocumentPosition.INTRO:
            if self.paragraph_index == 0 and self.sentence_index == 0:
                parts.append("This opens the entire document - establish the main topic engagingly")
        elif self.document_position == DocumentPosition.CONCLUSION:
            if self.sentence_index == self.sentence_total - 1:
                parts.append("This is the final sentence - provide a memorable conclusion")

        # Paragraph role guidance
        if self.paragraph_role == ParagraphRole.THESIS:
            parts.append("This paragraph states the main argument")
        elif self.paragraph_role == ParagraphRole.EVIDENCE:
            parts.append("Support with concrete evidence or examples")
        elif self.paragraph_role == ParagraphRole.COUNTERPOINT:
            parts.append("Acknowledge and address a potential objection")

        # Sentence role guidance
        if self.sentence_role == SentenceRole.TOPIC:
            parts.append("State the paragraph's main point clearly")
        elif self.sentence_role == SentenceRole.EXAMPLE:
            parts.append("Provide a specific, concrete example")
        elif self.sentence_role == SentenceRole.CONCLUSION:
            parts.append("Conclude this paragraph's argument")
        elif self.sentence_role == SentenceRole.TRANSITION:
            parts.append("Smoothly bridge to the next point")

        return " | ".join(parts)


@dataclass
class ParagraphPlan:
    """Plan for generating a full paragraph."""

    position: int
    total_paragraphs: int
    document_position: DocumentPosition
    role: ParagraphRole
    discourse_relation: DiscourseRelation

    propositions: List[Proposition]
    sentence_roles: List[SentenceRole]
    sentence_relations: List[DiscourseRelation]
    reference_types: List[ReferenceType]

    previous_paragraph_topic: Optional[str] = None
    next_paragraph_hint: Optional[str] = None


class StyleAwareGenerator:
    """Generates text with discourse awareness and style matching."""

    def __init__(
        self,
        corpus_paragraphs: List[str],
        author_name: str = "the author",
        strictness: float = 0.5,
        max_repair_attempts: int = 3,
    ):
        """Initialize generator.

        Args:
            corpus_paragraphs: Author's corpus for style extraction.
            author_name: Name for prompts.
            strictness: Style matching strictness (0-1).
            max_repair_attempts: Max regeneration attempts.
        """
        self.author_name = author_name
        self.strictness = strictness
        self.max_repair_attempts = max_repair_attempts

        # Initialize components
        self.discourse_analyzer = DiscourseAnalyzer()
        self.fingerprint_extractor = StyleFingerprintExtractor()

        # Extract style fingerprint
        logger.info(f"Extracting style fingerprint from {len(corpus_paragraphs)} paragraphs")
        self.fingerprint = self.fingerprint_extractor.extract(corpus_paragraphs)

        # Derive constraints
        self.constraints = self.fingerprint_extractor.derive_constraints(
            self.fingerprint, strictness
        )

        # Initialize verifier
        self.verifier = StyleVerifier(self.fingerprint, threshold=1.5)

        logger.info(f"Style fingerprint extracted:")
        logger.info(f"  - Sentence length: {self.fingerprint.sentence_length_mean:.1f} ± {self.fingerprint.sentence_length_std:.1f}")
        logger.info(f"  - Preferred transitions: {self.fingerprint.preferred_transitions[:5]}")
        logger.info(f"  - Distinctive vocabulary: {list(self.fingerprint.distinctive_vocabulary.keys())[:10]}")

    def plan_document(
        self,
        input_paragraphs: List[str],
    ) -> List[ParagraphPlan]:
        """Create generation plan for a document.

        Args:
            input_paragraphs: Source paragraphs with propositions.

        Returns:
            List of ParagraphPlans.
        """
        plans = []
        n_paras = len(input_paragraphs)

        for i, para in enumerate(input_paragraphs):
            # Determine document position
            if i == 0:
                doc_pos = DocumentPosition.INTRO
            elif i == n_paras - 1:
                doc_pos = DocumentPosition.CONCLUSION
            else:
                doc_pos = DocumentPosition.BODY

            # Determine paragraph role
            role = self._infer_paragraph_role(para, doc_pos, i, n_paras)

            # Determine relation to previous
            if i == 0:
                relation = DiscourseRelation.NONE
            else:
                relation = self._infer_paragraph_relation(para, i)

            # Extract propositions (simplified - would use actual extraction)
            sentences = split_into_sentences(para)
            propositions = [self._extract_proposition(s) for s in sentences]

            # Plan sentence roles and relations
            sent_roles = []
            sent_relations = []
            ref_types = []
            n_sents = len(sentences)

            for j in range(n_sents):
                # Sentence role
                if j == 0:
                    sent_roles.append(SentenceRole.TOPIC)
                    sent_relations.append(DiscourseRelation.NONE)
                    ref_types.append(ReferenceType.NONE if i == 0 else ReferenceType.BACKWARD)
                elif j == n_sents - 1:
                    sent_roles.append(SentenceRole.CONCLUSION)
                    sent_relations.append(DiscourseRelation.RESULT)
                    ref_types.append(ReferenceType.BACKWARD)
                else:
                    sent_roles.append(SentenceRole.SUPPORT)
                    sent_relations.append(DiscourseRelation.ELABORATION)
                    ref_types.append(ReferenceType.BACKWARD)

            plans.append(ParagraphPlan(
                position=i,
                total_paragraphs=n_paras,
                document_position=doc_pos,
                role=role,
                discourse_relation=relation,
                propositions=propositions,
                sentence_roles=sent_roles,
                sentence_relations=sent_relations,
                reference_types=ref_types,
                previous_paragraph_topic=plans[-1].propositions[0].subject if plans else None,
            ))

        return plans

    def build_generation_context(
        self,
        proposition: Proposition,
        paragraph_plan: ParagraphPlan,
        sentence_index: int,
        previous_sentence: Optional[str] = None,
        document_thesis: Optional[str] = None,
    ) -> GenerationContext:
        """Build full context for generating one sentence.

        Args:
            proposition: What to express.
            paragraph_plan: Plan for this paragraph.
            sentence_index: Index within paragraph.
            previous_sentence: Previous generated sentence.
            document_thesis: Main thesis of document.

        Returns:
            GenerationContext with all information for LLM.
        """
        return GenerationContext(
            proposition=proposition,
            document_position=paragraph_plan.document_position,
            paragraph_index=paragraph_plan.position,
            paragraph_total=paragraph_plan.total_paragraphs,
            sentence_index=sentence_index,
            sentence_total=len(paragraph_plan.propositions),
            paragraph_role=paragraph_plan.role,
            sentence_role=paragraph_plan.sentence_roles[sentence_index],
            discourse_relation=paragraph_plan.sentence_relations[sentence_index],
            reference_type=paragraph_plan.reference_types[sentence_index],
            previous_sentence=previous_sentence,
            previous_paragraph_topic=paragraph_plan.previous_paragraph_topic,
            document_thesis=document_thesis,
            style_constraints=self.constraints,
        )

    def verify_and_suggest_repairs(
        self,
        generated_text: str,
    ) -> Tuple[bool, StyleVerification, List[str]]:
        """Verify generated text and suggest repairs if needed.

        Args:
            generated_text: The generated text.

        Returns:
            Tuple of (is_acceptable, verification, repair_suggestions)
        """
        verification = self.verifier.verify(generated_text)

        if verification.is_acceptable:
            return True, verification, []

        repairs = self.verifier.suggest_repairs(generated_text, verification)
        return False, verification, repairs

    def build_repair_prompt(
        self,
        original_context: GenerationContext,
        failed_text: str,
        verification: StyleVerification,
        repairs: List[str],
    ) -> str:
        """Build a prompt for repairing failed generation.

        Args:
            original_context: Original generation context.
            failed_text: The text that failed verification.
            verification: Why it failed.
            repairs: Suggested repairs.

        Returns:
            Repair prompt for LLM.
        """
        lines = []

        lines.append("The previous generation did not match the author's style.")
        lines.append("")
        lines.append(f"Previous attempt: \"{failed_text}\"")
        lines.append("")
        lines.append("Issues detected:")
        for issue in verification.issues:
            lines.append(f"  - {issue}")
        lines.append("")
        lines.append("Required changes:")
        for repair in repairs:
            lines.append(f"  - {repair}")
        lines.append("")
        lines.append("Regenerate with these corrections:")
        lines.append("")
        lines.append(original_context.to_prompt(self.author_name))

        return "\n".join(lines)

    def get_style_summary(self) -> Dict:
        """Get a summary of the extracted style for debugging."""
        return {
            "author": self.author_name,
            "sentence_length": {
                "mean": round(self.fingerprint.sentence_length_mean, 1),
                "std": round(self.fingerprint.sentence_length_std, 1),
                "range": (self.fingerprint.sentence_length_min, self.fingerprint.sentence_length_max),
            },
            "paragraph_length_mean": round(self.fingerprint.paragraph_length_mean, 1),
            "burstiness": round(self.fingerprint.burstiness_mean, 3),
            "vocabulary_richness": round(self.fingerprint.type_token_ratio, 3),
            "preferred_transitions": self.fingerprint.preferred_transitions[:5],
            "distinctive_vocabulary": list(self.fingerprint.distinctive_vocabulary.keys())[:15],
            "framing_patterns": self.fingerprint.framing_patterns[:5],
            "characteristic_phrases": self.fingerprint.characteristic_phrases[:10],
        }

    def _infer_paragraph_role(
        self,
        paragraph: str,
        doc_pos: DocumentPosition,
        position: int,
        total: int,
    ) -> ParagraphRole:
        """Infer paragraph role from content and position."""
        para_lower = paragraph.lower()

        if doc_pos == DocumentPosition.INTRO:
            return ParagraphRole.THESIS if position == 0 else ParagraphRole.BACKGROUND
        if doc_pos == DocumentPosition.CONCLUSION:
            return ParagraphRole.CONCLUSION

        # Content-based
        if any(m in para_lower for m in ["for example", "for instance"]):
            return ParagraphRole.EVIDENCE
        if any(m in para_lower for m in ["however", "but", "on the other hand"]):
            return ParagraphRole.COUNTERPOINT

        return ParagraphRole.ELABORATION

    def _infer_paragraph_relation(self, paragraph: str, position: int) -> DiscourseRelation:
        """Infer relation to previous paragraph."""
        para_lower = paragraph.lower()
        first_words = para_lower[:100]

        for relation, markers in DISCOURSE_MARKERS.items():
            for marker in markers:
                if first_words.startswith(marker) or f" {marker}" in first_words:
                    return relation

        return DiscourseRelation.ELABORATION

    def _extract_proposition(self, sentence: str) -> Proposition:
        """Extract proposition from sentence (simplified)."""
        # This is a placeholder - actual implementation would use
        # proper SVO extraction from the existing pipeline
        words = sentence.split()
        return Proposition(
            subject=words[0] if words else "",
            predicate=words[1] if len(words) > 1 else "",
            object=" ".join(words[2:]) if len(words) > 2 else "",
        )


def create_generation_prompt(
    proposition: Proposition,
    context: GenerationContext,
    author_name: str,
) -> str:
    """Convenience function to create a full generation prompt.

    This is the main interface for the LLM caller.

    Args:
        proposition: What to express.
        context: Full generation context.
        author_name: Author to emulate.

    Returns:
        Complete prompt string.
    """
    return context.to_prompt(author_name)
