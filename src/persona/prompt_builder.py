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
from typing import List, Optional, Dict, TYPE_CHECKING
from functools import lru_cache

from .config import PersonaConfig

if TYPE_CHECKING:
    from ..rag.structural_grafter import GraftingGuidance


# =============================================================================
# Persona File Loading
# =============================================================================

@lru_cache(maxsize=4)
def _load_persona_file(persona_filename: str) -> Dict[str, any]:
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
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    filepath = prompts_dir / persona_filename

    result = {
        "narrative_frames": [],
        "conceptual_frames": [],
    }

    if not filepath.exists():
        # Try default
        default_path = prompts_dir / "default_persona.txt"
        if default_path.exists():
            filepath = default_path
        else:
            return result

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


def _get_worldview_filename(adapter_path: str = None) -> str:
    """Get worldview filename from config.json for a specific adapter.

    Args:
        adapter_path: Path to the adapter. If provided, looks up worldview
                     from that adapter's config. If None, uses first configured adapter.

    Returns:
        Worldview filename (e.g., "lovecraft_worldview.txt") or default.
    """
    try:
        from ..config import get_adapter_config, load_config

        if adapter_path:
            # Get worldview for specific adapter
            adapter_config = get_adapter_config(adapter_path)
            if adapter_config.worldview:
                return adapter_config.worldview
        else:
            # Fall back to first configured adapter's worldview
            config = load_config()
            if config.generation.lora_adapters:
                first_adapter = next(iter(config.generation.lora_adapters.values()))
                if first_adapter.worldview:
                    return first_adapter.worldview
    except Exception:
        pass
    return "default_persona.txt"


def _get_persona_frame(is_narrative: bool) -> str:
    """Get a persona frame from the configured worldview file."""
    filename = _get_worldview_filename()
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
        return "You are explaining a complex system. Document your understanding with precision."


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

# STRUCTURAL CONSTRAINTS (70% each) - Key style markers
# These push for the structural patterns that distinguish human writing
STRUCTURAL_CONSTRAINTS = [
    "Use em-dashes (—) for asides that reveal your inner turmoil.",
    "Vary sentence lengths dramatically. Follow a long sentence with a short one.",
]

# ROTATING (one random, 40%) - stylistic variety
ROTATING_CONSTRAINTS = [
    "Use fragments. Interrupt yourself.",
    "Let ideas collide without transition words.",
    "Do not explain. Imply.",
    "Use at least one rhetorical question.",
    "Start with a conjunction (But, And, Yet, So).",
    "End on an image or action, not a summary.",
]


def _build_constraints() -> str:
    """Build constraint block matching training format.

    Training used tiered constraints:
    - ALWAYS_CONSTRAINTS: 100%
    - FREQUENT_CONSTRAINTS: 70% each
    - STRUCTURAL_CONSTRAINTS: 70% each (em-dashes, varied lengths)
    - ROTATING_CONSTRAINTS: one random, 40%

    Total: typically 3-5 constraints
    """
    constraints = []

    # ALWAYS constraints (100%)
    constraints.extend(ALWAYS_CONSTRAINTS)

    # FREQUENT constraints (70% each) - match training
    for constraint in FREQUENT_CONSTRAINTS:
        if random.random() < 0.70:
            constraints.append(constraint)

    # STRUCTURAL constraints (70% each) - key style markers
    for constraint in STRUCTURAL_CONSTRAINTS:
        if random.random() < 0.70:
            constraints.append(constraint)

    # ROTATING constraints (one random, 40%)
    if random.random() < 0.40:
        constraints.append(random.choice(ROTATING_CONSTRAINTS))

    return "\n".join(f"[CONSTRAINT]: {c}" for c in constraints)


def _detect_content_type(content: str) -> bool:
    """Detect if content is narrative (events/story) or conceptual (explanation).

    Returns True for narrative, False for conceptual.
    """
    # Narrative indicators
    narrative_words = ['happened', 'went', 'came', 'saw', 'heard', 'felt',
                       'told', 'said', 'found', 'discovered', 'witnessed',
                       'arrived', 'left', 'began', 'ended', 'then', 'after']

    # Conceptual indicators
    conceptual_words = ['means', 'because', 'therefore', 'system', 'process',
                        'mechanism', 'function', 'works', 'causes', 'results',
                        'explains', 'defines', 'consists', 'involves']

    content_lower = content.lower()
    narrative_score = sum(1 for w in narrative_words if w in content_lower)
    conceptual_score = sum(1 for w in conceptual_words if w in content_lower)

    return narrative_score > conceptual_score


def build_persona_prompt(
    content: str,
    author: str,
    persona: PersonaConfig,
    vocabulary_palette: Optional[List[str]] = None,
    structural_guidance: Optional[str] = None,
    grafting_guidance: Optional['GraftingGuidance'] = None,
    target_words: Optional[int] = None,
    deterministic_constraints: bool = False,
    expand_for_texture: bool = False,
) -> str:
    """Build a prompt matching the training format EXACTLY.

    CRITICAL: This format MUST mirror training to activate LoRA weights correctly.
    The model was trained on simple prompts, not complex instruction-heavy ones.

    Persona frames are loaded from the file specified in config.json lora.worldview.

    Training format (from generate_flat_training.py line 1475):
    ```
    {persona_frame}

    Write approximately {word_count} words.

    Follow this structure: {skeleton}  (50% of training data)

    [CONSTRAINT]: Do not use: 'Moreover'...
    [CONSTRAINT]: Do not hedge...

    {neutral_text}
    ###
    ```

    That's IT. No worldview, no transformation directives, no style hints.
    """
    # Detect content type to select appropriate frame
    is_narrative = _detect_content_type(content)

    # Get situational persona frame from config file
    persona_frame = _get_persona_frame(is_narrative)

    # Calculate target words if not provided
    if target_words is None:
        target_words = len(content.split())

    # Build prompt parts
    parts = []

    # 1. Persona frame (REQUIRED - this is what triggers the LoRA)
    parts.append(persona_frame)
    parts.append("")  # blank line

    # 2. Word count (REQUIRED)
    parts.append(f"Write approximately {target_words} words.")

    # 2b. Expansion instruction (optional - encourages texture/flourishes)
    if expand_for_texture:
        parts.append("")
        parts.append(f"IMPORTANT: You MUST write {target_words} words. The input is shorter—you must EXPAND it substantially. "
                     "Add atmospheric details, sensory impressions, philosophical asides, and rhetorical flourishes. "
                     "Elaborate on implications. Add texture through vivid metaphors and characteristic digressions. "
                     "Do not simply paraphrase the input—amplify and enrich it to reach the full word count.")

    # 3. Skeleton structure (if available from grafting - matches training's 50% skeleton)
    if grafting_guidance and hasattr(grafting_guidance, 'skeleton') and grafting_guidance.skeleton:
        parts.append("")
        parts.append(f"Follow this structure: {grafting_guidance.skeleton}")

    # 4. Structural RAG guidance (rhythm patterns from corpus)
    # Format as simple guidance, not as "[CRITICAL - AUTHOR STYLE PATTERNS]" blocks
    if structural_guidance:
        parts.append("")
        # Extract just the key patterns, keep it concise
        parts.append(structural_guidance)

    # 5. Constraints (TIERED - matching training distribution)
    parts.append("")
    if deterministic_constraints:
        # For testing: include all constraints
        constraints = ALWAYS_CONSTRAINTS + FREQUENT_CONSTRAINTS + STRUCTURAL_CONSTRAINTS + [ROTATING_CONSTRAINTS[0]]
        parts.append("\n".join(f"[CONSTRAINT]: {c}" for c in constraints))
    else:
        parts.append(_build_constraints())

    # 6. Content
    parts.append("")
    parts.append(content)

    # 7. Stop token
    parts.append("###")

    return "\n".join(parts)
