"""Extract rhetorical skeletons from text.

A skeleton captures the argumentative structure without the content:
- [Concrete Analogy] → [Paradox] → [Rhetorical Question]
- [Observation] → [Counter-example] → [Conclusion]

This allows grafting an author's logical flow onto new content.
"""

from dataclasses import dataclass
from typing import List, Optional
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)

SKELETON_SYSTEM_PROMPT = '''You are a rhetorical analyst. You identify the structural moves in argumentative text.
Output ONLY a bracketed sequence like: [Move 1] -> [Move 2] -> [Move 3]
No explanation, no other text.'''

SKELETON_USER_PROMPT = '''Analyze the rhetorical structure of this text. Ignore the topic.

TEXT: "{text}"

Common moves: Concrete Analogy, Abstract Claim, Observation, Rhetorical Question, Paradox,
Counter-argument, Evidence, Qualification, Conclusion, Definition, Example, Implication,
Micro/Macro Contrast, Recursive Self-Reference, Direct Address, Building Tension, Resolution

Output ONLY the bracketed sequence:'''


@dataclass
class ArgumentSkeleton:
    """Rhetorical skeleton extracted from text."""
    moves: List[str]  # e.g., ["Concrete Analogy", "Paradox", "Rhetorical Question"]
    raw: str          # The raw LLM output

    def format_for_prompt(self) -> str:
        """Format skeleton for prompt injection."""
        if not self.moves:
            return self.raw
        return " → ".join(f"[{move}]" for move in self.moves)

    def to_metadata(self) -> str:
        """Format for storage as ChromaDB metadata."""
        return self.format_for_prompt()

    @classmethod
    def from_metadata(cls, metadata_str: str) -> 'ArgumentSkeleton':
        """Reconstruct from ChromaDB metadata."""
        moves = parse_skeleton_moves(metadata_str)
        return cls(moves=moves, raw=metadata_str)


def parse_skeleton_moves(text: str) -> List[str]:
    """Parse bracketed moves from skeleton string.

    Input: "[Concrete Analogy] -> [Paradox] -> [Rhetorical Question]"
    Output: ["Concrete Analogy", "Paradox", "Rhetorical Question"]
    """
    # Find all bracketed content
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches if m.strip()]


def extract_skeleton(text: str, llm_provider) -> ArgumentSkeleton:
    """Extract rhetorical skeleton from text using LLM.

    Args:
        text: The text to analyze.
        llm_provider: LLM provider with call() method.

    Returns:
        ArgumentSkeleton with extracted moves.
    """
    user_prompt = SKELETON_USER_PROMPT.format(text=text[:1500])  # Truncate long text

    try:
        response = llm_provider.call(
            system_prompt=SKELETON_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=200,
        )
        response = response.strip()

        # Parse moves from response
        moves = parse_skeleton_moves(response)

        if not moves:
            # Try to salvage if LLM returned unbracketed text
            # Split by common separators
            for sep in [' -> ', ' → ', ', ', '; ']:
                if sep in response:
                    parts = response.split(sep)
                    moves = [p.strip().strip('[]') for p in parts if p.strip()]
                    break

        if moves:
            logger.debug(f"Extracted skeleton: {moves}")
        else:
            logger.warning(f"Could not parse skeleton from: {response[:100]}")

        return ArgumentSkeleton(moves=moves, raw=response)

    except Exception as e:
        logger.error(f"Skeleton extraction failed: {e}")
        return ArgumentSkeleton(moves=[], raw="")


def extract_skeleton_batch(
    texts: List[str],
    llm_provider,
    show_progress: bool = True
) -> List[ArgumentSkeleton]:
    """Extract skeletons from multiple texts.

    Args:
        texts: List of texts to analyze.
        llm_provider: LLM provider.
        show_progress: Whether to show progress bar.

    Returns:
        List of ArgumentSkeletons.
    """
    skeletons = []

    if show_progress:
        try:
            from tqdm import tqdm
            texts = tqdm(texts, desc="Extracting skeletons")
        except ImportError:
            pass

    for text in texts:
        skeleton = extract_skeleton(text, llm_provider)
        skeletons.append(skeleton)

    return skeletons
