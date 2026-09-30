"""Semantic fidelity validation using DeepSeek.

Replaces the multi-step post-processing pipeline (semantic verification,
repair loop, grammar correction, repetition reduction) with a single
DeepSeek call that validates semantic equivalence and fixes only genuine
errors while preserving the restyled text's voice and structure.
"""

import json
from dataclasses import dataclass, field
from typing import List

from ..utils.logging import get_logger
from ..utils.prompts import load_prompt

logger = get_logger(__name__)

# Below this fraction of the restyled word count, a result is malformed rather
# than edited — cutting a hallucinated sentence lands well above it.
MIN_RESULT_WORD_RATIO = 0.5


@dataclass
class FidelityResult:
    """Result of semantic fidelity validation."""
    original: str
    corrected: str
    changes: List[dict] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return len(self.changes) > 0


def validate_semantic_fidelity(
    original: str,
    restyled: str,
    critic_provider,
) -> FidelityResult:
    """Validate and minimally correct restyled text for semantic fidelity.

    Uses the critic LLM to compare the restyled text against the original,
    fixing only genuine semantic errors (missing facts, reversed meaning,
    broken grammar) while preserving the restyled voice and structure.

    Args:
        original: The original source text (ground truth for meaning).
        restyled: The restyled text to validate.
        critic_provider: LLM provider for the validation call.

    Returns:
        FidelityResult with the corrected text and list of changes made.
    """
    system_prompt = load_prompt("semantic_fidelity")
    user_prompt = f"ORIGINAL:\n{original}\n\nRESTYLED:\n{restyled}"

    # The reply walks every claim of the original, quotes the restyled spans
    # and repeats the whole paragraph; 4x the restyled words cut half the
    # replies off mid-JSON, leaving those paragraphs unchecked.
    budget = max(4096, (len(original.split()) + 2 * len(restyled.split())) * 3)
    try:
        result = None
        for attempt in range(2):
            response = critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=budget * (attempt + 1),
                require_json=True,
            )
            try:
                result = json.loads(response)
                break
            except json.JSONDecodeError:
                if attempt == 1:
                    raise
                logger.info("Fidelity reply was cut off or malformed; retrying with more room")
        changes = result.get("changes", [])
        corrected = result.get("result", restyled)

        if not isinstance(corrected, str) or not corrected.strip():
            corrected = restyled
            changes = []
            logger.warning("Semantic fidelity returned empty/null result, keeping original restyled text")
        elif len(corrected.split()) < len(restyled.split()) * MIN_RESULT_WORD_RATIO:
            # The critic echoed the schema placeholder, summarised what it did, or
            # truncated the paragraph. Legitimate repairs never shrink it this far.
            logger.warning(
                f"Semantic fidelity result too short "
                f"({len(corrected.split())} vs {len(restyled.split())} words), "
                "discarding and keeping restyled text"
            )
            corrected = restyled
            changes = []

        if changes and isinstance(changes, list):
            for change in changes:
                if isinstance(change, dict):
                    issue = change.get("issue", "?")
                    kind = change.get("type")
                    logger.info(
                        f"Semantic fix [{kind}]: {issue}" if kind else f"Semantic fix: {issue}"
                    )

        return FidelityResult(
            original=original,
            corrected=corrected,
            changes=changes,
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse fidelity response: {e}")
        return FidelityResult(original=original, corrected=restyled)
    except Exception as e:
        logger.warning(f"Semantic fidelity check failed: {e}")
        return FidelityResult(original=original, corrected=restyled)
