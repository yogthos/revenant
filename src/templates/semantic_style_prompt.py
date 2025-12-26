"""Semantic-preserving style transfer prompts.

Key principle: STYLE from author, CONTENT from source.

The prompt must:
1. Preserve the semantic meaning of the source proposition
2. Apply the author's structural patterns (abstracted)
3. Use the author's general vocabulary preferences
4. NOT inject topic-specific terms from the author's domain

Example:
- Source: "All physical objects eventually deteriorate"
- Author style: Mao (uses "thus", "therefore", long complex sentences)
- Result: "It is a fundamental truth that all physical objects contain within
           themselves the seeds of their own deterioration, and thus the process
           of decay is inherent to material existence."

NOT:
- "Marxists hold that physical objects engage in class struggle" (wrong - injects Mao's topics)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum

from .style_vocabulary import (
    VocabularyClassifier,
    TemplateAbstractor,
    ClassifiedVocabulary,
    AbstractTemplate,
    TransferableStyle,
    UNIVERSAL_TRANSITIONS,
)
from .discourse import (
    SentenceContext,
    ParagraphContext,
    SentenceRole,
    DiscourseRelation,
    DocumentPosition,
    DISCOURSE_MARKERS,
)
from .filler import Proposition
from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SourceContent:
    """Content extracted from source text to be preserved."""

    # Core proposition
    proposition: Proposition

    # Key terms that MUST appear in output (from source)
    required_terms: Set[str]

    # Technical terms that should not be altered
    technical_terms: Set[str]

    # The logical relationship being expressed
    logical_structure: str  # e.g., "cause-effect", "contrast", "enumeration"

    # Original sentence for reference
    original: str


@dataclass
class AuthorStyle:
    """Style elements extracted from author (content-agnostic)."""

    # Sentence rhythm
    target_length: tuple  # (min, max) words
    burstiness: float

    # Transferable vocabulary
    transitions: List[str]
    general_verbs: List[str]
    general_adjectives: List[str]
    general_nouns: List[str]

    # Abstracted structural patterns
    patterns: List[AbstractTemplate]

    # Author name (for prompt)
    author_name: str


@dataclass
class SemanticStylePrompt:
    """A prompt that preserves source meaning while applying author style."""

    # What to express (from SOURCE)
    source_content: SourceContent

    # How to express it (from AUTHOR)
    author_style: AuthorStyle

    # Structural template to follow (abstracted)
    template: Optional[AbstractTemplate]

    # Discourse context
    sentence_role: SentenceRole
    discourse_relation: DiscourseRelation
    previous_sentence: Optional[str]
    paragraph_position: DocumentPosition

    def to_prompt(self) -> str:
        """Generate the complete prompt."""
        lines = []

        # Clear instruction about the task
        lines.append("=== TASK: STYLE TRANSFER ===")
        lines.append(f"Rewrite the source content in the style of {self.author_style.author_name}.")
        lines.append("")
        lines.append("CRITICAL RULES:")
        lines.append("1. PRESERVE the meaning and key terms from the source")
        lines.append("2. APPLY the author's sentence structure and rhythm")
        lines.append("3. USE the author's transition words and general vocabulary")
        lines.append("4. DO NOT add concepts or terms not present in the source")
        lines.append("")

        # Source content (what MUST be expressed)
        lines.append("=== SOURCE CONTENT (preserve this meaning) ===")
        lines.append(f"Original: \"{self.source_content.original}\"")
        lines.append("")
        lines.append("Core proposition:")
        lines.append(f"  Subject: {self.source_content.proposition.subject}")
        lines.append(f"  Predicate: {self.source_content.proposition.predicate}")
        if self.source_content.proposition.object:
            lines.append(f"  Object: {self.source_content.proposition.object}")

        if self.source_content.required_terms:
            lines.append(f"\nTerms that MUST appear: {', '.join(self.source_content.required_terms)}")

        if self.source_content.technical_terms:
            lines.append(f"Technical terms (do not alter): {', '.join(self.source_content.technical_terms)}")

        if self.source_content.logical_structure:
            lines.append(f"Logical structure: {self.source_content.logical_structure}")
        lines.append("")

        # Structural template (how to structure the sentence)
        if self.template:
            lines.append("=== SENTENCE STRUCTURE (from author) ===")
            lines.append(f"Pattern: {self.template.abstract_skeleton}")
            if self.template.structural_words:
                lines.append(f"Use structural words: {', '.join(self.template.structural_words)}")
            lines.append("")
            lines.append("Fill placeholders with SOURCE content:")
            for placeholder, desc in self.template.placeholder_types.items():
                lines.append(f"  {placeholder}: {desc}")
            lines.append("")

        # Author's style markers (transferable)
        lines.append("=== AUTHOR'S STYLE (apply these) ===")
        lines.append(f"Sentence length: {self.author_style.target_length[0]}-{self.author_style.target_length[1]} words")

        if self.author_style.transitions:
            lines.append(f"Transition preferences: {', '.join(self.author_style.transitions[:7])}")

        if self.author_style.general_verbs:
            lines.append(f"Preferred verbs: {', '.join(self.author_style.general_verbs[:10])}")

        if self.author_style.general_adjectives:
            lines.append(f"Preferred adjectives: {', '.join(self.author_style.general_adjectives[:10])}")

        if self.author_style.general_nouns:
            lines.append(f"Preferred abstract nouns: {', '.join(self.author_style.general_nouns[:10])}")
        lines.append("")

        # Discourse context
        lines.append("=== DISCOURSE CONTEXT ===")
        lines.append(f"Position: {self.paragraph_position.value} paragraph")
        lines.append(f"Sentence function: {self.sentence_role.value}")

        if self.discourse_relation != DiscourseRelation.NONE:
            lines.append(f"Relation to previous: {self.discourse_relation.value}")
            markers = DISCOURSE_MARKERS.get(self.discourse_relation, [])
            if markers:
                lines.append(f"Consider starting with: {', '.join(markers[:4])}")

        if self.previous_sentence:
            lines.append(f"Previous sentence: \"{self.previous_sentence[:80]}...\"")
        lines.append("")

        # Final instruction
        lines.append("=== OUTPUT ===")
        lines.append("Generate ONE sentence that:")
        lines.append("- Expresses the source proposition completely")
        lines.append("- Follows the structural pattern")
        lines.append("- Uses the author's vocabulary preferences where natural")
        lines.append("- Fits the discourse context")
        lines.append("")
        lines.append("Write ONLY the sentence:")

        return "\n".join(lines)


class SemanticStyleTransfer:
    """Orchestrates semantic-preserving style transfer."""

    def __init__(self, corpus_paragraphs: List[str], author_name: str):
        """Initialize with author corpus.

        Args:
            corpus_paragraphs: Author's writing samples.
            author_name: Author's name for prompts.
        """
        self.author_name = author_name
        self._nlp = None

        # Classify vocabulary
        logger.info(f"Classifying vocabulary from {len(corpus_paragraphs)} paragraphs")
        classifier = VocabularyClassifier()
        self.vocabulary = classifier.classify_corpus(corpus_paragraphs)

        # Abstract templates from corpus
        self.abstractor = TemplateAbstractor()
        self.abstract_templates = self._extract_abstract_templates(corpus_paragraphs)

        # Compute style metrics
        self.style = self._compute_style(corpus_paragraphs)

        logger.info(f"Extracted {len(self.abstract_templates)} abstract templates")
        logger.info(f"Transferable transitions: {list(self.vocabulary.transitions.keys())[:10]}")

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def _extract_abstract_templates(self, paragraphs: List[str]) -> List[AbstractTemplate]:
        """Extract and abstract templates from corpus."""
        templates = []

        for para in paragraphs[:20]:  # Limit for efficiency
            sentences = split_into_sentences(para)
            for sent in sentences:
                # Create a simple skeleton (this is simplified)
                # In full implementation, would use the existing extractor
                skeleton = self._create_skeleton(sent)
                if skeleton:
                    template = self.abstractor.abstract_template(skeleton, sent)
                    templates.append(template)

        return templates

    def _create_skeleton(self, sentence: str) -> Optional[str]:
        """Create skeleton from sentence (simplified)."""
        doc = self.nlp(sentence)

        # Find main clause structure
        parts = []
        for token in doc:
            if token.dep_ == "ROOT":
                # Get subject
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        parts.append("[SUBJECT]")
                        break
                parts.append(token.text)
                # Get object/complement
                for child in token.children:
                    if child.dep_ in ("dobj", "attr", "ccomp"):
                        parts.append("[OBJECT]")
                        break
                break

        if len(parts) >= 2:
            return " ".join(parts)
        return None

    def _compute_style(self, paragraphs: List[str]) -> AuthorStyle:
        """Compute transferable style metrics."""
        all_sentences = []
        for para in paragraphs:
            all_sentences.extend(split_into_sentences(para))

        lengths = [len(s.split()) for s in all_sentences]
        mean_len = sum(lengths) / len(lengths) if lengths else 20
        std_len = (sum((l - mean_len) ** 2 for l in lengths) / len(lengths)) ** 0.5 if lengths else 5

        return AuthorStyle(
            target_length=(max(5, int(mean_len - std_len)), int(mean_len + std_len)),
            burstiness=std_len / mean_len if mean_len > 0 else 0,
            transitions=sorted(self.vocabulary.transitions.keys(),
                              key=lambda x: self.vocabulary.transitions[x],
                              reverse=True)[:15],
            general_verbs=sorted(self.vocabulary.general_verbs.keys(),
                                key=lambda x: self.vocabulary.general_verbs[x],
                                reverse=True)[:20],
            general_adjectives=sorted(self.vocabulary.general_adjectives.keys(),
                                     key=lambda x: self.vocabulary.general_adjectives[x],
                                     reverse=True)[:20],
            general_nouns=sorted(self.vocabulary.general_nouns.keys(),
                                key=lambda x: self.vocabulary.general_nouns[x],
                                reverse=True)[:20],
            patterns=self.abstract_templates[:10],
            author_name=self.author_name,
        )

    def extract_source_content(
        self,
        sentence: str,
        technical_terms: Optional[Set[str]] = None,
    ) -> SourceContent:
        """Extract content from source sentence.

        Args:
            sentence: Source sentence.
            technical_terms: Known technical terms to preserve.

        Returns:
            SourceContent with proposition and required terms.
        """
        doc = self.nlp(sentence)

        # Extract SVO
        subject = ""
        predicate = ""
        obj = ""

        for token in doc:
            if token.dep_ == "ROOT":
                predicate = token.text
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = " ".join([t.text for t in child.subtree])
                    elif child.dep_ in ("dobj", "attr", "ccomp", "acomp"):
                        obj = " ".join([t.text for t in child.subtree])

        # Identify required terms (nouns and key verbs from source)
        required = set()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                required.add(token.text.lower())
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                required.add(token.lemma_.lower())

        # Detect logical structure
        logical = "assertion"  # default
        sent_lower = sentence.lower()
        if any(w in sent_lower for w in ["because", "since", "due to"]):
            logical = "cause-effect"
        elif any(w in sent_lower for w in ["but", "however", "although"]):
            logical = "contrast"
        elif any(w in sent_lower for w in ["therefore", "thus", "consequently"]):
            logical = "conclusion"
        elif any(w in sent_lower for w in ["if", "when", "unless"]):
            logical = "conditional"

        return SourceContent(
            proposition=Proposition(subject=subject, predicate=predicate, object=obj),
            required_terms=required,
            technical_terms=technical_terms or set(),
            logical_structure=logical,
            original=sentence,
        )

    def select_template(
        self,
        source_content: SourceContent,
        sentence_role: SentenceRole,
    ) -> Optional[AbstractTemplate]:
        """Select best matching abstract template.

        Args:
            source_content: The source content to express.
            sentence_role: Role of sentence in paragraph.

        Returns:
            Best matching abstract template or None.
        """
        # For now, simple selection based on role
        # In full implementation, would match logical structure

        for template in self.abstract_templates:
            if template.pattern_type:
                if sentence_role == SentenceRole.TOPIC and template.pattern_type == "AUTHORITY_CLAIM":
                    return template
                if sentence_role == SentenceRole.CONCLUSION and template.pattern_type == "RESULT":
                    return template

        # Return first available template
        return self.abstract_templates[0] if self.abstract_templates else None

    def build_prompt(
        self,
        source_sentence: str,
        sentence_context: SentenceContext,
        paragraph_context: ParagraphContext,
        technical_terms: Optional[Set[str]] = None,
    ) -> SemanticStylePrompt:
        """Build a semantic-preserving style transfer prompt.

        Args:
            source_sentence: Source sentence to transfer.
            sentence_context: Sentence-level context.
            paragraph_context: Paragraph-level context.
            technical_terms: Technical terms to preserve exactly.

        Returns:
            SemanticStylePrompt ready for LLM.
        """
        # Extract source content
        source_content = self.extract_source_content(source_sentence, technical_terms)

        # Select appropriate template
        template = self.select_template(source_content, sentence_context.role)

        return SemanticStylePrompt(
            source_content=source_content,
            author_style=self.style,
            template=template,
            sentence_role=sentence_context.role,
            discourse_relation=sentence_context.discourse_relation,
            previous_sentence=sentence_context.previous_topic,
            paragraph_position=paragraph_context.document_position,
        )
