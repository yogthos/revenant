"""Global context analysis for documents."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ..models.graph import DocumentGraph, SemanticGraph
from ..models.style import StyleProfile
from ..llm.provider import LLMProvider
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GlobalContext:
    """Document-level context for style transfer.

    This context is set once at the start of document processing
    and persists throughout generation.
    """
    # Document content
    thesis: str
    intent: str  # persuade, inform, narrate, explain
    keywords: List[str]
    perspective: str  # first_person_singular, first_person_plural, third_person

    # Style target
    style_dna: str
    author_name: str
    target_burstiness: float
    target_sentence_length: float
    top_vocabulary: List[str]

    # Processing state
    total_paragraphs: int = 0
    processed_paragraphs: int = 0
    generated_summary: str = ""  # Summary of previously generated text

    def to_system_prompt(self) -> str:
        """Convert context to LLM system prompt.

        Returns:
            System prompt string with document context.
        """
        return f"""You are a skilled writer adapting text to match a specific author's style.

TARGET STYLE: {self.author_name}
{self.style_dna}

DOCUMENT CONTEXT:
- Thesis: {self.thesis}
- Intent: {self.intent}
- Keywords to incorporate: {', '.join(self.keywords[:10])}
- Perspective: {self.perspective}

STATISTICAL TARGETS:
- Average sentence length: ~{self.target_sentence_length:.0f} words
- Rhythm variation (burstiness): {self.target_burstiness:.2f}
- Vocabulary preferences: {', '.join(self.top_vocabulary[:15])}

RULES:
1. Preserve ALL semantic meaning from the source
2. Match the target style's sentence rhythm and vocabulary
3. Maintain the {self.perspective} perspective throughout
4. Use transition words and phrases characteristic of the target style
5. Do not add new information not present in the source"""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "thesis": self.thesis,
            "intent": self.intent,
            "keywords": self.keywords,
            "perspective": self.perspective,
            "style_dna": self.style_dna,
            "author_name": self.author_name,
            "target_burstiness": self.target_burstiness,
            "target_sentence_length": self.target_sentence_length,
            "top_vocabulary": self.top_vocabulary,
            "total_paragraphs": self.total_paragraphs,
            "processed_paragraphs": self.processed_paragraphs,
        }


@dataclass
class ParagraphContext:
    """Context for a specific paragraph being processed.

    Set once per paragraph, contains paragraph-specific info
    plus reference to global context.
    """
    paragraph_idx: int
    role: str  # INTRO, BODY, CONCLUSION
    semantic_graph: SemanticGraph
    previous_summary: str  # Summary of previously generated paragraphs
    sentence_count_target: int
    total_propositions: int

    def to_prompt_section(self) -> str:
        """Convert to prompt section for paragraph context.

        Returns:
            Prompt section string.
        """
        # Summarize the semantic graph
        props = [f"- {node.text}" for node in self.semantic_graph.nodes[:5]]
        props_text = "\n".join(props)
        if len(self.semantic_graph.nodes) > 5:
            props_text += f"\n  ... and {len(self.semantic_graph.nodes) - 5} more propositions"

        return f"""PARAGRAPH {self.paragraph_idx + 1} ({self.role}):
Previous context: {self.previous_summary or 'This is the first paragraph.'}

Semantic content to express:
{props_text}

Target: ~{self.sentence_count_target} sentences"""


class GlobalContextAnalyzer:
    """Analyzes documents to extract global context for generation.

    Creates the GlobalContext object that persists throughout
    document processing, enabling coherent multi-paragraph generation.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize analyzer.

        Args:
            llm_provider: Optional LLM for enhanced analysis.
        """
        self.llm_provider = llm_provider

    def analyze(
        self,
        document_graph: DocumentGraph,
        style_profile: StyleProfile
    ) -> GlobalContext:
        """Analyze document and style to create global context.

        Args:
            document_graph: Semantic graph of the document.
            style_profile: Target style profile.

        Returns:
            GlobalContext instance.
        """
        # Extract thesis (use document graph's or generate)
        thesis = document_graph.thesis
        if not thesis and document_graph.paragraphs:
            thesis = self._extract_thesis_from_graph(document_graph)

        # Get style profile effective values
        author = style_profile.primary_author

        context = GlobalContext(
            thesis=thesis,
            intent=document_graph.intent,
            keywords=document_graph.keywords,
            perspective=document_graph.perspective,
            style_dna=style_profile.get_effective_style_dna(),
            author_name=style_profile.get_author_name(),
            target_burstiness=style_profile.get_effective_burstiness(),
            target_sentence_length=style_profile.get_effective_avg_sentence_length(),
            top_vocabulary=style_profile.get_effective_vocab(),
            total_paragraphs=len(document_graph.paragraphs),
            processed_paragraphs=0
        )

        logger.info(
            f"Created global context: intent={context.intent}, "
            f"target_author={context.author_name}, "
            f"paragraphs={context.total_paragraphs}"
        )

        return context

    def _extract_thesis_from_graph(self, document_graph: DocumentGraph) -> str:
        """Extract thesis from semantic graph when not available.

        Args:
            document_graph: Document's semantic graph.

        Returns:
            Thesis string.
        """
        if not document_graph.paragraphs:
            return ""

        # Get first paragraph (intro)
        intro = document_graph.paragraphs[0]
        if intro.nodes:
            # Use the first proposition or two
            thesis_parts = [n.text for n in intro.nodes[:2]]
            return " ".join(thesis_parts)

        return ""

    def create_paragraph_context(
        self,
        paragraph_graph: SemanticGraph,
        global_context: GlobalContext,
        previous_paragraphs: List[str]
    ) -> ParagraphContext:
        """Create context for a specific paragraph.

        Args:
            paragraph_graph: Semantic graph for the paragraph.
            global_context: Global document context.
            previous_paragraphs: Previously generated paragraph texts.

        Returns:
            ParagraphContext instance.
        """
        # Summarize previous paragraphs
        if previous_paragraphs:
            summary = self._summarize_paragraphs(previous_paragraphs)
        else:
            summary = ""

        # Calculate target sentence count
        # Based on proposition count and target sentence length
        prop_count = len(paragraph_graph.nodes)
        avg_props_per_sentence = 1.5  # Rough estimate
        sentence_target = max(1, int(prop_count / avg_props_per_sentence))

        return ParagraphContext(
            paragraph_idx=paragraph_graph.paragraph_idx,
            role=paragraph_graph.role.value,
            semantic_graph=paragraph_graph,
            previous_summary=summary,
            sentence_count_target=sentence_target,
            total_propositions=prop_count
        )

    def _summarize_paragraphs(self, paragraphs: List[str]) -> str:
        """Create a brief summary of previous paragraphs.

        Args:
            paragraphs: List of paragraph texts.

        Returns:
            Summary string.
        """
        if not paragraphs:
            return ""

        # Simple approach: take first sentence of each paragraph
        summaries = []
        for para in paragraphs[-3:]:  # Last 3 paragraphs
            sentences = para.split('.')
            if sentences:
                first_sent = sentences[0].strip()
                if first_sent:
                    summaries.append(first_sent)

        if len(paragraphs) > 3:
            prefix = f"[{len(paragraphs) - 3} earlier paragraphs...] "
        else:
            prefix = ""

        return prefix + ". ".join(summaries) + "."

    def update_context_after_paragraph(
        self,
        global_context: GlobalContext,
        generated_paragraph: str
    ) -> None:
        """Update global context after generating a paragraph.

        Args:
            global_context: Context to update.
            generated_paragraph: The just-generated paragraph.
        """
        global_context.processed_paragraphs += 1

        # Update summary
        if global_context.generated_summary:
            global_context.generated_summary += " " + self._get_first_sentence(generated_paragraph)
        else:
            global_context.generated_summary = self._get_first_sentence(generated_paragraph)

    def _get_first_sentence(self, text: str) -> str:
        """Get first sentence from text.

        Args:
            text: Text to extract from.

        Returns:
            First sentence.
        """
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip() + "."
        return text[:100] + "..."
