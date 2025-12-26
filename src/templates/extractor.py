"""Skeleton extraction from sentences using spaCy."""

import hashlib
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger
from .models import (
    SentenceTemplate,
    TemplateSlot,
    SlotType,
    SentenceType,
    RhetoricalRole,
    LogicalRelation,
)
from .statistics import SentenceClassifier

logger = get_logger(__name__)


@dataclass
class ExtractedSlot:
    """Intermediate representation of an extracted slot."""
    text: str
    start: int
    end: int
    slot_type: SlotType
    pos_tags: List[str]
    dep_label: str
    head_text: str


class SkeletonExtractor:
    """Extracts syntactic skeletons from sentences.

    Converts sentences like:
        "The cosmos reveals its secrets through patient observation."
    Into skeletons like:
        "The [SUBJECT] [VERB] its [OBJECT] through [MODIFIER] [NOUN]."

    With slot definitions for filling with new content.
    """

    def __init__(self):
        self._nlp = None
        self.classifier = SentenceClassifier()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract_template(
        self,
        sentence: str,
        position: int = 0,
        total_sentences: int = 1,
        author: str = "",
        document_id: str = ""
    ) -> SentenceTemplate:
        """Extract a template from a single sentence.

        Args:
            sentence: The sentence to extract from.
            position: Position in paragraph (0 = first).
            total_sentences: Total sentences in paragraph.
            author: Author name for metadata.
            document_id: Document ID for metadata.

        Returns:
            SentenceTemplate with skeleton and slots.
        """
        logger.debug(f"Extracting template from: '{sentence[:50]}...' (pos={position}/{total_sentences})")
        doc = self.nlp(sentence)

        # Extract slots
        slots = self._extract_slots(doc)

        # Build skeleton by replacing content with placeholders
        skeleton = self._build_skeleton(sentence, slots)

        # Extract POS pattern
        pos_pattern = " ".join([token.pos_ for token in doc])

        # Compute complexity
        complexity = self._compute_complexity(doc)

        # Count clauses
        clause_count = self._count_clauses(doc)

        # Classify sentence
        sentence_type = self._get_sentence_type(position, total_sentences)
        rhetorical_role = self.classifier.classify_role(sentence, position, total_sentences)
        logical_relation = self.classifier.classify_relation(sentence)

        # Generate ID
        template_id = self._generate_id(sentence, author)

        # Convert slots to TemplateSlot objects
        template_slots = [
            TemplateSlot(
                name=f"{slot.slot_type.value}_{i}",
                slot_type=slot.slot_type,
                position=slot.start,
                required=True,
                pos_tags=slot.pos_tags,
            )
            for i, slot in enumerate(slots)
        ]

        template = SentenceTemplate(
            id=template_id,
            skeleton=skeleton,
            pos_pattern=pos_pattern,
            word_count=len(doc),
            complexity_score=complexity,
            clause_count=clause_count,
            sentence_type=sentence_type,
            rhetorical_role=rhetorical_role,
            logical_relation=logical_relation,
            original_text=sentence,
            slots=template_slots,
            author=author,
            document_id=document_id,
        )

        logger.debug(
            f"Extracted template: skeleton='{skeleton[:40]}...', "
            f"slots={len(template_slots)}, role={rhetorical_role.value}, "
            f"complexity={complexity:.1f}"
        )

        return template

    def extract_templates_from_paragraph(
        self,
        paragraph: str,
        author: str = "",
        document_id: str = ""
    ) -> List[SentenceTemplate]:
        """Extract templates from all sentences in a paragraph.

        Args:
            paragraph: The paragraph text.
            author: Author name.
            document_id: Document ID.

        Returns:
            List of SentenceTemplates.
        """
        sentences = split_into_sentences(paragraph)
        templates = []

        for i, sent in enumerate(sentences):
            try:
                template = self.extract_template(
                    sent,
                    position=i,
                    total_sentences=len(sentences),
                    author=author,
                    document_id=document_id
                )
                templates.append(template)
            except Exception as e:
                logger.warning(f"Failed to extract template from: {sent[:50]}... - {e}")

        return templates

    def _extract_slots(self, doc) -> List[ExtractedSlot]:
        """Extract content slots from a parsed sentence.

        Identifies:
        - Subject noun phrases
        - Main verbs
        - Object noun phrases
        - Prepositional phrases
        - Subordinate clauses
        - Modifiers (adjectives, adverbs)
        """
        slots = []

        # Find the root verb
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if not root:
            return slots

        # Extract subject
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subject_span = self._get_subtree_span(token, doc)
                if subject_span:
                    slots.append(ExtractedSlot(
                        text=subject_span[0],
                        start=subject_span[1],
                        end=subject_span[2],
                        slot_type=SlotType.SUBJECT,
                        pos_tags=self._get_pos_tags(subject_span[0]),
                        dep_label=token.dep_,
                        head_text=token.head.text,
                    ))

        # Extract main verb phrase
        verb_span = self._get_verb_phrase(root, doc)
        if verb_span:
            slots.append(ExtractedSlot(
                text=verb_span[0],
                start=verb_span[1],
                end=verb_span[2],
                slot_type=SlotType.VERB,
                pos_tags=self._get_pos_tags(verb_span[0]),
                dep_label="ROOT",
                head_text="",
            ))

        # Extract objects
        for token in doc:
            if token.dep_ in ("dobj", "pobj", "attr"):
                obj_span = self._get_subtree_span(token, doc)
                if obj_span and not self._overlaps_existing(obj_span, slots):
                    slots.append(ExtractedSlot(
                        text=obj_span[0],
                        start=obj_span[1],
                        end=obj_span[2],
                        slot_type=SlotType.OBJECT,
                        pos_tags=self._get_pos_tags(obj_span[0]),
                        dep_label=token.dep_,
                        head_text=token.head.text,
                    ))

        # Extract prepositional phrases
        for token in doc:
            if token.dep_ == "prep":
                prep_span = self._get_prep_phrase(token, doc)
                if prep_span and not self._overlaps_existing(prep_span, slots):
                    slots.append(ExtractedSlot(
                        text=prep_span[0],
                        start=prep_span[1],
                        end=prep_span[2],
                        slot_type=SlotType.PREPOSITIONAL,
                        pos_tags=self._get_pos_tags(prep_span[0]),
                        dep_label=token.dep_,
                        head_text=token.head.text,
                    ))

        # Extract subordinate clauses
        for token in doc:
            if token.dep_ in ("advcl", "relcl", "ccomp", "xcomp"):
                clause_span = self._get_clause_span(token, doc)
                if clause_span and not self._overlaps_existing(clause_span, slots):
                    slots.append(ExtractedSlot(
                        text=clause_span[0],
                        start=clause_span[1],
                        end=clause_span[2],
                        slot_type=SlotType.CLAUSE,
                        pos_tags=self._get_pos_tags(clause_span[0]),
                        dep_label=token.dep_,
                        head_text=token.head.text,
                    ))

        # Sort by position
        slots.sort(key=lambda s: s.start)

        return slots

    def _get_subtree_span(self, token, doc) -> Optional[Tuple[str, int, int]]:
        """Get the text span covered by a token's subtree."""
        subtree = list(token.subtree)
        if not subtree:
            return None

        start = min(t.idx for t in subtree)
        end = max(t.idx + len(t.text) for t in subtree)
        text = doc.text[start:end]

        # Skip if too short or just punctuation
        if len(text.strip()) < 2 or text.strip() in ".,;:!?":
            return None

        return (text, start, end)

    def _get_verb_phrase(self, root, doc) -> Optional[Tuple[str, int, int]]:
        """Get the main verb phrase."""
        # Include auxiliaries and particles
        verb_tokens = [root]

        for child in root.children:
            if child.dep_ in ("aux", "auxpass", "neg", "prt"):
                verb_tokens.append(child)

        if not verb_tokens:
            return None

        verb_tokens.sort(key=lambda t: t.i)
        start = verb_tokens[0].idx
        end = verb_tokens[-1].idx + len(verb_tokens[-1].text)
        text = doc.text[start:end]

        return (text, start, end)

    def _get_prep_phrase(self, prep_token, doc) -> Optional[Tuple[str, int, int]]:
        """Get a prepositional phrase."""
        # Get the preposition and its object
        tokens = [prep_token]

        for child in prep_token.subtree:
            tokens.append(child)

        if len(tokens) < 2:
            return None

        start = min(t.idx for t in tokens)
        end = max(t.idx + len(t.text) for t in tokens)
        text = doc.text[start:end]

        return (text, start, end)

    def _get_clause_span(self, clause_root, doc) -> Optional[Tuple[str, int, int]]:
        """Get a subordinate clause span."""
        subtree = list(clause_root.subtree)
        if len(subtree) < 3:  # Clauses should have at least 3 words
            return None

        start = min(t.idx for t in subtree)
        end = max(t.idx + len(t.text) for t in subtree)
        text = doc.text[start:end]

        return (text, start, end)

    def _overlaps_existing(
        self,
        span: Tuple[str, int, int],
        slots: List[ExtractedSlot]
    ) -> bool:
        """Check if a span overlaps with existing slots."""
        _, start, end = span

        for slot in slots:
            # Check for overlap
            if not (end <= slot.start or start >= slot.end):
                return True

        return False

    def _get_pos_tags(self, text: str) -> List[str]:
        """Get POS tags for a text span."""
        doc = self.nlp(text)
        return [token.pos_ for token in doc]

    def _build_skeleton(
        self,
        sentence: str,
        slots: List[ExtractedSlot]
    ) -> str:
        """Build skeleton by replacing slots with placeholders."""
        if not slots:
            return sentence

        # Sort slots by position (reverse to replace from end)
        sorted_slots = sorted(slots, key=lambda s: s.start, reverse=True)

        skeleton = sentence
        for i, slot in enumerate(sorted_slots):
            placeholder = f"[{slot.slot_type.value.upper()}]"
            skeleton = skeleton[:slot.start] + placeholder + skeleton[slot.end:]

        return skeleton

    def _compute_complexity(self, doc) -> float:
        """Compute syntactic complexity as max dependency depth."""
        max_depth = 0

        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 20:
                    break
            max_depth = max(max_depth, depth)

        return float(max_depth)

    def _count_clauses(self, doc) -> int:
        """Count the number of clauses in the sentence."""
        clause_deps = {"advcl", "relcl", "ccomp", "xcomp", "acl", "ROOT"}
        count = 0

        for token in doc:
            if token.dep_ in clause_deps:
                count += 1

        return max(1, count)  # At least 1 (main clause)

    def _get_sentence_type(self, position: int, total: int) -> SentenceType:
        """Determine sentence type based on position."""
        if position == 0:
            return SentenceType.OPENER
        elif position == total - 1:
            return SentenceType.CLOSER
        else:
            return SentenceType.BODY

    def _generate_id(self, sentence: str, author: str) -> str:
        """Generate unique ID for a template."""
        content = f"{author}:{sentence}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class TemplateLibrary:
    """Collection of templates from an author's corpus."""

    def __init__(self, author: str):
        self.author = author
        self.templates: List[SentenceTemplate] = []
        self.extractor = SkeletonExtractor()

    def add_paragraph(self, paragraph: str, document_id: str = "") -> int:
        """Add templates from a paragraph.

        Returns:
            Number of templates added.
        """
        new_templates = self.extractor.extract_templates_from_paragraph(
            paragraph, self.author, document_id
        )
        self.templates.extend(new_templates)
        return len(new_templates)

    def add_corpus(self, paragraphs: List[str], document_id: str = "") -> int:
        """Add templates from entire corpus.

        Returns:
            Total templates added.
        """
        total = 0
        for i, para in enumerate(paragraphs):
            doc_id = f"{document_id}_{i}" if document_id else str(i)
            total += self.add_paragraph(para, doc_id)

        logger.info(f"Extracted {total} templates from {len(paragraphs)} paragraphs")
        return total

    def get_by_type(
        self,
        sentence_type: SentenceType = None,
        rhetorical_role: RhetoricalRole = None,
        logical_relation: LogicalRelation = None
    ) -> List[SentenceTemplate]:
        """Filter templates by classification."""
        result = self.templates

        if sentence_type:
            result = [t for t in result if t.sentence_type == sentence_type]

        if rhetorical_role:
            result = [t for t in result if t.rhetorical_role == rhetorical_role]

        if logical_relation:
            result = [t for t in result if t.logical_relation == logical_relation]

        return result

    def get_by_complexity(
        self,
        min_complexity: float = 0,
        max_complexity: float = 10
    ) -> List[SentenceTemplate]:
        """Filter templates by complexity range."""
        return [
            t for t in self.templates
            if min_complexity <= t.complexity_score <= max_complexity
        ]

    def get_by_length(
        self,
        min_length: int = 0,
        max_length: int = 100
    ) -> List[SentenceTemplate]:
        """Filter templates by word count range."""
        return [
            t for t in self.templates
            if min_length <= t.word_count <= max_length
        ]
