"""Structural Grafter - retrieves author samples and skeletons for grafting.

Instead of just copying rhythm patterns, this retrieves:
1. A semantically similar sample from the author's corpus
2. The rhetorical skeleton of that sample

The model then "grafts" the skeleton's logic onto the new content.
"""

from dataclasses import dataclass
from typing import Optional

from .corpus_indexer import get_indexer
from .skeleton_extractor import ArgumentSkeleton, extract_skeleton
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GraftingGuidance:
    """Guidance for structural grafting."""
    sample_text: str           # The retrieved author sample
    skeleton: ArgumentSkeleton # The rhetorical skeleton

    def format_for_prompt(self) -> str:
        """Format guidance for prompt injection."""
        skeleton_str = self.skeleton.format_for_prompt()

        return f'''[STYLE SAMPLE (BLUEPRINT)]:
"{self.sample_text}"

[RHETORICAL SKELETON (YOUR GUIDE)]:
{skeleton_str}

INSTRUCTION: Follow the argumentative structure from the skeleton.
Copy the LOGIC FLOW and SENTENCE RHYTHM, not the words.'''


class StructuralGrafter:
    """Retrieves samples and skeletons for structural grafting."""

    def __init__(self, author: str, llm_provider=None):
        """Initialize the grafter.

        Args:
            author: Author name to retrieve samples from.
            llm_provider: Optional LLM provider for on-the-fly skeleton extraction.
                         If not provided, only pre-computed skeletons are used.
        """
        self.author = author
        self.indexer = get_indexer()
        self.llm_provider = llm_provider
        self._loaded = False

    def get_grafting_guidance(self, input_text: str) -> Optional[GraftingGuidance]:
        """Get grafting guidance for input text.

        Retrieves the most semantically similar sample from the author's
        corpus and returns it with its rhetorical skeleton.

        Args:
            input_text: The input text to find a matching sample for.

        Returns:
            GraftingGuidance or None if no suitable sample found.
        """
        # Retrieve similar chunks
        try:
            similar = self.indexer.retrieve_similar(
                author=self.author,
                query_text=input_text,
                n=1
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve similar chunks: {e}")
            return None

        if not similar:
            logger.debug(f"No similar samples found for {self.author}")
            return None

        # Get best match
        best = similar[0]
        sample_text = best["text"]
        skeleton_str = best.get("skeleton", "")

        # Build skeleton
        if skeleton_str:
            # Use pre-computed skeleton
            skeleton = ArgumentSkeleton.from_metadata(skeleton_str)
            logger.debug(f"Using pre-computed skeleton: {skeleton.format_for_prompt()}")
        elif self.llm_provider:
            # Extract skeleton on-the-fly
            logger.debug("Extracting skeleton on-the-fly")
            skeleton = extract_skeleton(sample_text, self.llm_provider)
        else:
            # No skeleton available
            logger.warning("No skeleton available and no LLM provider for extraction")
            skeleton = ArgumentSkeleton(moves=[], raw="")

        if not skeleton.moves:
            logger.debug("Skeleton has no moves, skipping grafting")
            return None

        return GraftingGuidance(
            sample_text=sample_text,
            skeleton=skeleton
        )


# Cache for grafter instances
_grafter_cache = {}


def get_structural_grafter(author: str, llm_provider=None) -> StructuralGrafter:
    """Get or create a structural grafter for an author.

    Args:
        author: Author name.
        llm_provider: Optional LLM provider for on-the-fly extraction.

    Returns:
        StructuralGrafter instance.
    """
    cache_key = author
    if cache_key not in _grafter_cache:
        _grafter_cache[cache_key] = StructuralGrafter(author, llm_provider)
    return _grafter_cache[cache_key]
