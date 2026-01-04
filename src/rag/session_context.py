"""Session context for Style RAG.

Manages RAG context for a style transfer session, providing
cached style examples to inject into prompts.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ..utils.logging import get_logger
from .style_retriever import StyleRetriever, RetrievedChunk, get_retriever

logger = get_logger(__name__)


@dataclass
class StyleRAGContext:
    """RAG context for a style transfer session.

    Holds retrieved style examples to inject into prompts.
    Examples are retrieved once and cached for the session.
    """

    author: str
    examples: List[str] = field(default_factory=list)
    _retriever: Optional[StyleRetriever] = field(default=None, repr=False)
    num_examples: int = 3
    use_diverse: bool = True

    def __post_init__(self):
        """Initialize retriever if not provided."""
        if self._retriever is None:
            self._retriever = get_retriever()

    def load_examples(self, sample_text: str) -> int:
        """Load style examples based on sample text.

        Retrieves examples from the author's corpus that match
        the structural and semantic characteristics of the sample.

        Args:
            sample_text: Sample of text to match (e.g., first paragraph).

        Returns:
            Number of examples loaded.
        """
        logger.info(f"Loading {self.num_examples} style examples for {self.author}")

        if self.use_diverse:
            chunks = self._retriever.retrieve_diverse(
                sample_text, self.author, k=self.num_examples
            )
        else:
            chunks = self._retriever.retrieve(
                sample_text, self.author, k=self.num_examples
            )

        self.examples = [chunk.text for chunk in chunks]

        logger.info(f"Loaded {len(self.examples)} style examples")
        return len(self.examples)

    def refresh_examples(self, new_text: str) -> int:
        """Refresh examples based on new text.

        Call this when moving to a new paragraph or section
        with different content.

        Args:
            new_text: New text to match.

        Returns:
            Number of examples refreshed.
        """
        return self.load_examples(new_text)

    def format_for_prompt(self) -> str:
        """Format examples for injection into prompt.

        Returns:
            Formatted string ready for prompt insertion.
        """
        if not self.examples:
            return ""

        lines = ["\nSTYLE REFERENCE (study vocabulary and rhythm, do NOT copy content):"]
        for i, example in enumerate(self.examples, 1):
            # Clean up example (trim to reasonable length)
            text = example.strip()
            if len(text) > 500:
                text = text[:497] + "..."

            lines.append(f"[Example {i}]: \"{text}\"")

        lines.append("")  # Empty line before content
        return "\n".join(lines)

    def has_examples(self) -> bool:
        """Check if examples have been loaded."""
        return len(self.examples) > 0

    @property
    def example_count(self) -> int:
        """Get number of loaded examples."""
        return len(self.examples)


class RAGContextManager:
    """Manages RAG context across transfer sessions."""

    def __init__(self, retriever: Optional[StyleRetriever] = None):
        """Initialize the manager.

        Args:
            retriever: StyleRetriever to use. If None, uses default.
        """
        self._retriever = retriever or get_retriever()
        self._contexts: dict[str, StyleRAGContext] = {}

    def get_context(
        self,
        author: str,
        sample_text: Optional[str] = None,
        num_examples: int = 3,
        use_diverse: bool = True,
    ) -> StyleRAGContext:
        """Get or create a RAG context for an author.

        Args:
            author: Author name.
            sample_text: Optional sample text to initialize with.
            num_examples: Number of examples to retrieve.
            use_diverse: Whether to use diverse retrieval.

        Returns:
            StyleRAGContext for the author.
        """
        if author not in self._contexts:
            context = StyleRAGContext(
                author=author,
                _retriever=self._retriever,
                num_examples=num_examples,
                use_diverse=use_diverse,
            )

            if sample_text:
                context.load_examples(sample_text)

            self._contexts[author] = context

        return self._contexts[author]

    def clear_context(self, author: str) -> None:
        """Clear cached context for an author.

        Args:
            author: Author name.
        """
        if author in self._contexts:
            del self._contexts[author]

    def clear_all(self) -> None:
        """Clear all cached contexts."""
        self._contexts.clear()


# Module singleton
_context_manager = None


def get_context_manager() -> RAGContextManager:
    """Get the RAG context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = RAGContextManager()
    return _context_manager


def create_rag_context(
    author: str,
    sample_text: Optional[str] = None,
    num_examples: int = 3,
) -> StyleRAGContext:
    """Convenience function to create a RAG context.

    Args:
        author: Author name.
        sample_text: Sample text to initialize with.
        num_examples: Number of examples.

    Returns:
        Initialized StyleRAGContext.
    """
    return get_context_manager().get_context(
        author=author,
        sample_text=sample_text,
        num_examples=num_examples,
    )
