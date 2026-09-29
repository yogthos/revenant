"""Structural Grafter - retrieves author samples and skeletons for grafting.

Instead of just copying rhythm patterns, this retrieves:
1. A semantically similar sample from the author's corpus
2. The rhetorical skeleton of that sample

The model then "grafts" the skeleton's logic onto the new content.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .corpus_indexer import get_indexer
from .skeleton_extractor import ArgumentSkeleton, extract_skeleton
from ..utils.logging import get_logger

logger = get_logger(__name__)

# How many candidates to look at when the best match has to be skipped.
_EXCLUDE_CANDIDATES = 5
# Share of the shorter text's words both texts have in common above which a
# corpus chunk counts as the excluded paragraph.
_SAME_TEXT_OVERLAP = 0.5


def _words(text: str) -> set:
    return set(re.findall(r"[a-z']+", text.lower()))


def _same_text(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) > _SAME_TEXT_OVERLAP


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
        self._skeleton_cache = {}

    def get_grafting_guidance(self, input_text: str, exclude: Optional[str] = None) -> Optional[GraftingGuidance]:
        """Get grafting guidance for input text.

        Retrieves the most semantically similar sample from the author's
        corpus and returns it with its rhetorical skeleton.

        Args:
            input_text: The input text to find a matching sample for.
            exclude: Text the sample must not be. Training passes the row's
                target so a row never grafts its own paragraph, which the
                model would never get at inference.

        Returns:
            GraftingGuidance or None if no suitable sample found.
        """
        # Retrieve similar chunks
        try:
            similar = self.indexer.retrieve_similar(
                author=self.author,
                query_text=input_text,
                n=_EXCLUDE_CANDIDATES if exclude else 1
            )
            if exclude:
                similar = [c for c in similar if not _same_text(c["text"], exclude)][:1]
                if not similar:
                    return None
        except Exception as e:
            logger.warning(f"Failed to retrieve similar chunks: {e}")
            return None

        if not similar:
            # Fall back to random chunks
            try:
                random_chunks = self.indexer.get_random_chunks(self.author, n=3)
                if random_chunks:
                    logger.debug(f"Using random chunk as fallback for grafting guidance")
                    similar = [{"text": random_chunks[0], "skeleton": ""}]
                else:
                    logger.debug(f"No samples found for {self.author}")
                    return None
            except Exception as e:
                logger.debug(f"Could not get random chunks for fallback: {e}")
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
            # Extract skeleton on-the-fly, once per sample
            if sample_text not in self._skeleton_cache:
                logger.debug("Extracting skeleton on-the-fly")
                self._skeleton_cache[sample_text] = extract_skeleton(sample_text, self.llm_provider)
            skeleton = self._skeleton_cache[sample_text]
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


# Cache key is (author, id(services)) so different Services containers get
# distinct grafters — otherwise an injected container would inherit the
# process-wide default's indexer from an earlier call.
_grafter_cache = {}


def get_structural_grafter(author: str, llm_provider=None) -> StructuralGrafter:
    """Get or create a structural grafter for an author.

    The cache is partitioned by the active Services container so an injected
    container does not see a StructuralGrafter built against the process-wide
    default (which would bypass DI).

    Args:
        author: Author name.
        llm_provider: Optional LLM provider for on-the-fly extraction.

    Returns:
        StructuralGrafter instance.
    """
    from ..services import get_default_services
    services = get_default_services()
    cache_key = (author, id(services))
    if cache_key not in _grafter_cache:
        _grafter_cache[cache_key] = StructuralGrafter(author, llm_provider)
    elif llm_provider is not None and _grafter_cache[cache_key].llm_provider is None:
        _grafter_cache[cache_key].llm_provider = llm_provider
    return _grafter_cache[cache_key]
