"""Build persona-injected prompts for subjective style transfer.

CRITICAL: The inference prompt format MUST match the training format exactly.
Training uses situational personas ("You are writing in a diary...") not identity
personas ("You are H.P. Lovecraft"). The model learned to respond to specific
persona frames during training, and we must trigger those same frames.

Training format (from generate_flat_training.py):
1. Situational persona frame (acting direction)
2. Word count constraint
3. Skeleton structure (if available, 50% of training data)
4. 2-4 [CONSTRAINT] blocks (tiered system)
5. Neutral input content
6. ### stop token

The model was NOT trained on:
- "CRITICAL: YOU ARE THIS AUTHOR" headers
- "TRANSFORMATION DIRECTIVE" sections
- 15+ constraints per prompt
- Style hints, vocabulary hints
- Any special formatting beyond the simple training format

CONFIGURATION:
- Persona frames are loaded from prompts/{lora.worldview} file
- File format uses sections: [WORLDVIEW], [PERSONA_FRAMES_NARRATIVE], [PERSONA_FRAMES_CONCEPTUAL]
- Frames are separated by '---' within each section
"""

import random
import re
from pathlib import Path
from typing import Any, Optional, Dict
from functools import lru_cache
from ..utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Persona File Loading
# =============================================================================

@lru_cache(maxsize=4)
def _load_persona_file(persona_filename: str) -> Dict[str, Any]:
    """Load and parse persona file from prompts folder.

    File format (must match training exactly):
    [PERSONA_FRAMES_NARRATIVE]
    Frame 1
    ---
    Frame 2

    [PERSONA_FRAMES_CONCEPTUAL]
    Frame 1
    ---
    Frame 2

    Returns dict with keys: narrative_frames, conceptual_frames
    """
    if not persona_filename:
        return {"narrative_frames": [], "conceptual_frames": []}

    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    filepath = prompts_dir / persona_filename

    result = {
        "narrative_frames": [],
        "conceptual_frames": [],
    }

    if not filepath.exists():
        raise FileNotFoundError(
            f"Persona file not found: {persona_filename}. "
            f"Check the 'worldview' setting in config.json — the filename must exist in prompts/."
        )

    content = filepath.read_text(encoding="utf-8")

    # Parse sections
    sections = re.split(r'\[([A-Z_]+)\]', content)

    current_section = None
    for part in sections:
        part = part.strip()
        if part in ("PERSONA_FRAMES_NARRATIVE", "PERSONA_FRAMES_CONCEPTUAL"):
            current_section = part
        elif current_section and part:
            if current_section == "PERSONA_FRAMES_NARRATIVE":
                frames = [f.strip() for f in part.split("---") if f.strip()]
                result["narrative_frames"] = frames
            elif current_section == "PERSONA_FRAMES_CONCEPTUAL":
                frames = [f.strip() for f in part.split("---") if f.strip()]
                result["conceptual_frames"] = frames

    return result


def _get_worldview_filename(adapter_path: Optional[str] = None) -> str:
    """Get worldview filename from config.json for a specific adapter or fused model.

    Args:
        adapter_path: Path to the adapter or fused model. If provided, looks up
                     the worldview from that entry's config (checking both
                     `lora_adapters` and `models`). If None, falls back to the
                     first enabled entry.

    Returns:
        Worldview filename (e.g., "lovecraft_worldview.txt") or default.
    """
    try:
        from ..config import get_adapter_config, get_fused_model_config, load_config

        if adapter_path:
            # Check LoRA adapters first
            adapter_config = get_adapter_config(adapter_path)
            if adapter_config.worldview:
                return adapter_config.worldview
            # Fall through to fused models (path may be a fused model instead)
            fused_config = get_fused_model_config(adapter_path)
            if fused_config.worldview:
                return fused_config.worldview
        else:
            # Fall back to first enabled entry's worldview
            config = load_config()
            for adapter_config in config.generation.lora_adapters.values():
                if adapter_config.enabled and adapter_config.worldview:
                    return adapter_config.worldview
            for fused_config in config.generation.models.values():
                if fused_config.enabled and fused_config.worldview:
                    return fused_config.worldview
    except Exception as e:
        logger.warning(f"Failed to determine worldview filename: {e}")
    return "default_persona.txt"


def _get_persona_frame(is_narrative: bool, adapter_path: Optional[str] = None) -> str:
    """Get a persona frame from the configured worldview file."""
    filename = _get_worldview_filename(adapter_path)
    persona_data = _load_persona_file(filename)

    if is_narrative:
        frames = persona_data.get("narrative_frames", [])
    else:
        frames = persona_data.get("conceptual_frames", [])

    if frames:
        return random.choice(frames)

    # Fallback
    if is_narrative:
        return "You are recounting events you witnessed firsthand. Describe what happened as if confessing to a close friend."
    else:
        return "State these facts with the absolute, pitiless precision of a machine."


# =============================================================================
# Tiered Constraints (Must match training)
# =============================================================================
# Training used a TIERED system with randomization.
# We must replicate the SAME distribution at inference.

# ALWAYS included (100%) - these are clear AI tells
ALWAYS_CONSTRAINTS = [
    "Do not use: 'Moreover', 'Furthermore', 'Therefore', 'Thus', 'Hence', 'In conclusion', 'It is important to note', 'It is worth noting', 'This highlights', 'This underscores', 'In essence', 'Ultimately'.",
    "Do not hedge. Avoid: 'arguably', 'it could be said', 'one might argue', 'perhaps it is', 'it seems that'. State things directly.",
]

# FREQUENT (70% each) - strong anti-patterns
FREQUENT_CONSTRAINTS = [
    "Do not start with a topic sentence. Start with a sensory detail, a question, or mid-thought.",
    "Do not use numbered lists or 'Firstly/Secondly/Thirdly' structures.",
]

# ROTATING (one random, 40%) - stylistic variety
# Must match training exactly (generate_flat_training.py ROTATING_CONSTRAINTS)
ROTATING_CONSTRAINTS = [
    "Use fragments. Interrupt yourself with dashes (—).",
    "Let ideas collide without transition words.",
    "Do not explain. Imply.",
    "Use at least one rhetorical question.",
    "Interrupt yourself with a parenthetical thought.",
    "Start the paragraph with a conjunction (But, And, Yet, So).",
    "Be biased. Be opinionated. Do not balance your argument.",
    "Vary sentence lengths dramatically. Follow a long sentence with a short one.",
    "Use concrete nouns instead of abstractions. Not 'the concept' but the thing itself.",
    "End on an image or action, not a summary.",
]


def _build_constraints() -> str:
    """Build constraint block matching training format.

    Training used tiered constraints (3 tiers only):
    - ALWAYS_CONSTRAINTS: 100%
    - FREQUENT_CONSTRAINTS: 70% each
    - ROTATING_CONSTRAINTS: one random, 40%

    Total: typically 3-4 constraints
    """
    constraints = []

    # ALWAYS constraints (100%)
    constraints.extend(ALWAYS_CONSTRAINTS)

    # FREQUENT constraints (70% each) - match training
    for constraint in FREQUENT_CONSTRAINTS:
        if random.random() < 0.70:
            constraints.append(constraint)

    # ROTATING constraints (one random, 40%)
    if random.random() < 0.40:
        constraints.append(random.choice(ROTATING_CONSTRAINTS))

    return "\n".join(f"[CONSTRAINT]: {c}" for c in constraints)


def _detect_content_type(content: str) -> bool:
    """Detect if content is narrative (events/story) or conceptual (explanation).

    Uses shared classifier to match training detection exactly.

    Returns True for narrative, False for conceptual.
    """
    from ..utils.content_classifier import is_narrative
    return is_narrative(content)


def build_persona_instruction(
    content: str,
    target_words: Optional[int] = None,
    deterministic_constraints: bool = False,
    adapter_path: Optional[str] = None,
) -> str:
    """Build the instruction half of a training-format prompt.

    Training rows keep the instruction and the neutral input in separate
    fields, which LlamaFactory joins into one user turn. Keeping them
    separate here lets the generator lay them out the same way.

    ``content`` is only used to pick a narrative or conceptual frame. Training
    classifies the neutral input too, so the frame matches.

    Nothing else goes in. Training rows never had RAG rhythm hints, and their
    "Follow this structure" skeletons described the target itself (half the
    rows have none), so a skeleton of some other paragraph is out of
    distribution.
    """
    is_narrative = _detect_content_type(content)

    # Situational persona frame from the config file (this is what triggers the LoRA)
    persona_frame = _get_persona_frame(is_narrative, adapter_path=adapter_path)

    if target_words is None:
        target_words = len(content.split())

    # Training format: "Write approximately N words." on its own line, nothing else
    parts = [persona_frame, "", f"Write approximately {target_words} words."]

    # Constraints (TIERED - matching training distribution)
    parts.append("")
    if deterministic_constraints:
        # For testing: include all constraints
        constraints = ALWAYS_CONSTRAINTS + FREQUENT_CONSTRAINTS + [ROTATING_CONSTRAINTS[0]]
        parts.append("\n".join(f"[CONSTRAINT]: {c}" for c in constraints))
    else:
        parts.append(_build_constraints())

    return "\n".join(parts)


def build_persona_prompt(
    content: str,
    target_words: Optional[int] = None,
    deterministic_constraints: bool = False,
    adapter_path: Optional[str] = None,
) -> str:
    """Build a flat prompt in the MLX training format:

    ```
    {persona_frame}

    Write approximately {word_count} words.

    [CONSTRAINT]: Do not use: 'Moreover'...

    {neutral_text}
    ###
    ```

    Chat-template generators should use build_persona_instruction() and pass
    the content separately instead.
    """
    instruction = build_persona_instruction(
        content,
        target_words=target_words,
        deterministic_constraints=deterministic_constraints,
        adapter_path=adapter_path,
    )
    return f"{instruction}\n\n{content}\n###"
