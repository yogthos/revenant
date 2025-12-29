"""Document-level context extraction for improved generation and critique.

Extracts global document context once at the start of processing to:
1. Help the critic understand the overall argument and catch inconsistencies
2. Provide minimal guidance to generation for better coherence
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json

from ..utils.logging import get_logger
from ..utils.nlp import get_nlp, extract_entities, extract_keywords

logger = get_logger(__name__)


@dataclass
class DocumentContext:
    """Global context extracted from a document.

    Used to maintain coherence across paragraphs and give the critic
    document-level awareness.
    """
    # Core document understanding
    thesis: str = ""  # One sentence summary of main argument/purpose
    intent: str = ""  # informative, persuasive, narrative, analytical, explanatory

    # Key elements to track
    key_entities: List[str] = field(default_factory=list)  # Important names, places, terms
    key_concepts: List[str] = field(default_factory=list)  # Main ideas/themes

    # Style guidance
    tone: str = ""  # formal, conversational, academic, poetic, etc.

    # Statistics
    total_paragraphs: int = 0
    total_words: int = 0

    def to_critic_context(self) -> str:
        """Format context for the critic prompt."""
        parts = []

        if self.thesis:
            parts.append(f"Document thesis: {self.thesis}")
        if self.intent:
            parts.append(f"Document intent: {self.intent}")
        if self.key_entities:
            parts.append(f"Key entities: {', '.join(self.key_entities[:10])}")
        if self.tone:
            parts.append(f"Tone: {self.tone}")

        return "\n".join(parts)

    def to_generation_hint(self) -> str:
        """Format minimal context for generation instruction.

        Keep this very brief to avoid confusing the base model.
        """
        if self.intent and self.tone:
            return f"{self.tone} {self.intent} text"
        elif self.intent:
            return f"{self.intent} text"
        elif self.tone:
            return f"{self.tone} text"
        return ""


class DocumentContextExtractor:
    """Extracts document context using LLM and NLP.

    Uses a combination of:
    1. LLM for thesis/intent extraction (requires understanding)
    2. spaCy for entities/keywords (fast, reliable)
    """

    def __init__(self, llm_provider=None):
        """Initialize extractor.

        Args:
            llm_provider: Optional LLM provider for thesis/intent extraction.
                         If None, uses heuristics only.
        """
        self.llm_provider = llm_provider
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(self, text: str) -> DocumentContext:
        """Extract document context from text.

        Args:
            text: Full document text.

        Returns:
            DocumentContext with extracted information.
        """
        context = DocumentContext()

        # Basic stats
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        context.total_paragraphs = len(paragraphs)
        context.total_words = len(text.split())

        # Extract entities and keywords using spaCy (fast, reliable)
        try:
            context.key_entities = extract_entities(text)[:15]
            context.key_concepts = extract_keywords(text, top_n=10)
        except Exception as e:
            logger.warning(f"Failed to extract entities/keywords: {e}")

        # Extract thesis/intent using LLM if available
        if self.llm_provider:
            try:
                self._extract_with_llm(text, context)
            except Exception as e:
                logger.warning(f"LLM context extraction failed: {e}")
                self._extract_with_heuristics(text, context)
        else:
            self._extract_with_heuristics(text, context)

        logger.info(
            f"Extracted document context: {context.total_paragraphs} paragraphs, "
            f"intent={context.intent}, tone={context.tone}"
        )

        return context

    def _extract_with_llm(self, text: str, context: DocumentContext) -> None:
        """Extract thesis and intent using LLM."""
        # Use first ~1000 words for context extraction (enough to understand document)
        words = text.split()
        sample = ' '.join(words[:1000]) if len(words) > 1000 else text

        system_prompt = """Analyze this text and extract:
1. thesis: One sentence summarizing the main argument or purpose
2. intent: One of: informative, persuasive, narrative, analytical, explanatory
3. tone: One of: formal, conversational, academic, poetic, technical, journalistic

Respond in JSON format only:
{"thesis": "...", "intent": "...", "tone": "..."}"""

        user_prompt = f"Text to analyze:\n\n{sample}"

        response = self.llm_provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=200,
        )

        # Parse JSON response
        try:
            # Handle markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            data = json.loads(response)
            context.thesis = data.get("thesis", "")
            context.intent = data.get("intent", "")
            context.tone = data.get("tone", "")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:100]}")
            self._extract_with_heuristics(text, context)

    def _extract_with_heuristics(self, text: str, context: DocumentContext) -> None:
        """Extract context using heuristics when LLM unavailable."""
        doc = self.nlp(text[:5000])  # Analyze first ~5000 chars

        # Detect tone based on vocabulary and structure
        formal_markers = {"therefore", "furthermore", "consequently", "thus", "hence"}
        academic_markers = {"research", "study", "analysis", "findings", "hypothesis"}
        conversational_markers = {"you", "we", "i", "let's", "here's"}

        text_lower = text.lower()

        if any(m in text_lower for m in academic_markers):
            context.tone = "academic"
        elif any(m in text_lower for m in formal_markers):
            context.tone = "formal"
        elif any(m in text_lower for m in conversational_markers):
            context.tone = "conversational"
        else:
            context.tone = "neutral"

        # Detect intent based on sentence patterns
        questions = sum(1 for sent in doc.sents if sent.text.strip().endswith("?"))
        imperatives = sum(1 for token in doc if token.dep_ == "ROOT" and token.tag_ == "VB")

        if questions > len(list(doc.sents)) * 0.2:
            context.intent = "explanatory"
        elif imperatives > 5:
            context.intent = "persuasive"
        else:
            context.intent = "informative"

        # Use first sentence as rough thesis
        sentences = list(doc.sents)
        if sentences:
            context.thesis = sentences[0].text.strip()[:200]


def extract_document_context(
    text: str,
    llm_provider=None,
) -> DocumentContext:
    """Convenience function to extract document context.

    Args:
        text: Full document text.
        llm_provider: Optional LLM provider for better extraction.

    Returns:
        DocumentContext with extracted information.
    """
    extractor = DocumentContextExtractor(llm_provider)
    return extractor.extract(text)
