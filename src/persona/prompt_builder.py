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
from typing import Any, Optional, Dict, TYPE_CHECKING
from functools import lru_cache
from ..utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..rag.structural_grafter import GraftingGuidance

# =============================================================================
# Persona File Loading
# =============================================================================

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _parse_directives(section: str, filename: str) -> list:
    directives = []
    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        check, _, text = line.partition(":")
        check, text = check.strip(), text.strip()
        if check not in DIRECTIVE_CHECKS or not text:
            raise ValueError(f"{filename}: unknown directive check {check!r}; "
                             f"known checks: {', '.join(sorted(DIRECTIVE_CHECKS))}")
        directives.append((check, text))
    return directives


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

    [DIRECTIVES]
    check_name: Directive text

    Returns dict with keys: narrative_frames, conceptual_frames, directives
    (a list of (check_name, text) pairs).
    """
    if not persona_filename:
        return {"narrative_frames": [], "conceptual_frames": [], "directives": []}

    filepath = _PROMPTS_DIR / persona_filename

    result = {
        "narrative_frames": [],
        "conceptual_frames": [],
        "directives": [],
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
        if part in ("PERSONA_FRAMES_NARRATIVE", "PERSONA_FRAMES_CONCEPTUAL", "DIRECTIVES"):
            current_section = part
        elif current_section and part:
            if current_section == "PERSONA_FRAMES_NARRATIVE":
                frames = [f.strip() for f in part.split("---") if f.strip()]
                result["narrative_frames"] = frames
            elif current_section == "PERSONA_FRAMES_CONCEPTUAL":
                frames = [f.strip() for f in part.split("---") if f.strip()]
                result["conceptual_frames"] = frames
            elif current_section == "DIRECTIVES":
                result["directives"] = _parse_directives(part, persona_filename)

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


def _get_persona_frame(is_narrative: bool, adapter_path: Optional[str] = None,
                       worldview: Optional[str] = None) -> str:
    """Get a persona frame from the worldview file (named, or from config)."""
    filename = worldview or _get_worldview_filename(adapter_path)
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


_TRANSITIONS = r"\b(however|moreover|furthermore|therefore|thus|hence|consequently|nevertheless|nonetheless)\b"

# How to tell whether a text obeys a constraint. Training keeps only the
# constraints its target obeys; a row that says "never use Therefore" above a
# target that does teaches the model to ignore constraints. Constraints with
# no check here ("Do not explain. Imply.") are never put on training rows.
_CONSTRAINT_CHECKS = {
    ALWAYS_CONSTRAINTS[0]: lambda t: not re.search(
        r"\b(moreover|furthermore|therefore|thus|hence|in conclusion|it is important to note|"
        r"it is worth noting|this highlights|this underscores|in essence|ultimately)\b", t, re.I),
    ALWAYS_CONSTRAINTS[1]: lambda t: not re.search(
        r"\b(arguably|it could be said|one might argue|perhaps it is|it seems that)\b", t, re.I),
    FREQUENT_CONSTRAINTS[1]: lambda t: not re.search(
        r"\b(firstly|secondly|thirdly)\b|^\s*\d+[.)]\s", t, re.I | re.M),
    "Use fragments. Interrupt yourself with dashes (—).": lambda t: bool(re.search(r"[—–]| -- ", t)),
    "Let ideas collide without transition words.": lambda t: not re.search(_TRANSITIONS, t, re.I),
    "Use at least one rhetorical question.": lambda t: "?" in t,
    "Interrupt yourself with a parenthetical thought.": lambda t: bool(re.search(r"\([^)]+\)", t)),
    "Start the paragraph with a conjunction (But, And, Yet, So).":
        lambda t: bool(re.match(r"[\s\"'“‘]*(but|and|yet|so)\b", t, re.I)),
}


def constraint_holds(constraint: str, text: str) -> bool:
    """True if ``text`` demonstrably obeys ``constraint``."""
    check = _CONSTRAINT_CHECKS.get(constraint)
    return bool(check and check(text))


# Style directives. The wording lives in the author's worldview file
# ([DIRECTIVES], "check: text"); these are the generic checks it can name.
# A directive only ever goes on a training row whose target obeys it, and at
# inference only when the grafted exemplar obeys it.
def _sentences(text: str) -> list:
    return [s for s in re.split(r'(?<=[.!?])["\'\u201d\u2019)]*\s+(?=["\'\u201c\u2018(]?[A-Z0-9])', text.strip())
            if s.strip()]


def _lengths(text: str) -> list:
    return [len(s.split()) for s in _sentences(text)] or [0]


DIRECTIVE_CHECKS = {
    "opens_long": lambda t: _lengths(t)[0] >= 30,
    "opens_short": lambda t: _lengths(t)[0] <= 10,
    "long_sentence": lambda t: max(_lengths(t)) >= 40,
    "short_after_long": lambda t: any(a >= 30 and b <= 12 for a, b in zip(_lengths(t), _lengths(t)[1:])),
    "ends_short": lambda t: len(_lengths(t)) > 1 and _lengths(t)[-1] <= 12,
    "semicolon": lambda t: ";" in t,
    "colon": lambda t: bool(re.search(r"\w:\s", t)),
    "parenthesis": lambda t: bool(re.search(r"\([^)]+\)", t)),
    "dash": lambda t: bool(re.search(r"[\u2014\u2013]| -- ", t)),
    "question": lambda t: "?" in t,
    "example": lambda t: bool(re.search(r"\b(for example|for instance|e\.g\.|take|consider)\b", t, re.I)),
    "hypothetical": lambda t: bool(re.search(r"\b(suppose|supposing|let us|imagine|if we)\b", t, re.I)),
    "not_but": lambda t: bool(re.search(r"\bnot\b[^.;:?!]{1,60}?\bbut\b", t)),
    "conjunction_start": lambda t: any(re.match(r"[\"\u201c]?(But|And|Yet|So|Or|Nor)\b", s)
                                       for s in _sentences(t)[1:]),
    "we": lambda t: bool(re.search(r"\b(we|us|our)\b", t, re.I)),
    "first_person": lambda t: bool(re.search(r"\bI\b", t)),
    "scare_quotes": lambda t: bool(re.search(r"[\"\u201c][^\"\u201d]{1,40}[\"\u201d]", t)),
    "concession": lambda t: bool(re.search(r"\b(of course|no doubt|doubtless|admittedly|it is true that)\b",
                                           t, re.I)),
}


def _get_directives(adapter_path: Optional[str] = None, worldview: Optional[str] = None) -> list:
    try:
        return _load_persona_file(worldview or _get_worldview_filename(adapter_path))["directives"]
    except FileNotFoundError:
        return []


def _build_directive_constraints(directives: list, exemplar: Optional[str]) -> str:
    """Generic constraints plus 2-4 directives, all true of ``exemplar``.

    ``exemplar`` is the row's target in training and the grafted corpus
    paragraph at inference. Without one only the generic constraints apply.
    """
    constraints = list(ALWAYS_CONSTRAINTS)
    chosen = []
    if exemplar is not None:
        constraints = [c for c in constraints if constraint_holds(c, exemplar)]
        obeyed = [text for check, text in directives if DIRECTIVE_CHECKS[check](exemplar)]
        chosen = random.sample(obeyed, min(len(obeyed), random.randint(2, 4)))
    return "\n".join(f"[CONSTRAINT]: {c}" for c in constraints + chosen)


def _format_constraints(constraints: list, satisfied_by: Optional[str] = None) -> str:
    if satisfied_by is not None:
        constraints = [c for c in constraints if constraint_holds(c, satisfied_by)]
    return "\n".join(f"[CONSTRAINT]: {c}" for c in constraints)


def _build_constraints(satisfied_by: Optional[str] = None) -> str:
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

    return _format_constraints(constraints, satisfied_by)


def _detect_content_type(content: str) -> bool:
    """Detect if content is narrative (events/story) or conceptual (explanation).

    Uses shared classifier to match training detection exactly.

    Returns True for narrative, False for conceptual.
    """
    from ..utils.content_classifier import is_narrative
    return is_narrative(content)


def build_persona_instruction(
    content: str,
    structural_guidance: Optional[str] = None,
    grafting_guidance: Optional['GraftingGuidance'] = None,
    target_words: Optional[int] = None,
    deterministic_constraints: bool = False,
    adapter_path: Optional[str] = None,
    worldview: Optional[str] = None,
    satisfied_by: Optional[str] = None,
) -> str:
    """Build the persona instruction for one paragraph.

    Training rows (filter_training_data.PersonaBuilder) and inference both
    build it here, so they get the same frame, guidance and constraints.

    ``content`` picks the narrative or conceptual frame. ``worldview`` names
    the persona file directly; otherwise it comes from the adapter's entry in
    config.json. ``satisfied_by`` is a training row's target: constraints it
    doesn't obey are left out. At inference the grafted exemplar plays that
    part for authors whose worldview file has [DIRECTIVES].
    """
    is_narrative = _detect_content_type(content)

    # Situational persona frame from the worldview file (this is what triggers the LoRA)
    persona_frame = _get_persona_frame(is_narrative, adapter_path=adapter_path, worldview=worldview)

    if target_words is None:
        target_words = len(content.split())

    parts = [persona_frame, "", f"Write approximately {target_words} words."]

    # Rhetorical skeleton of the most similar corpus paragraph
    if grafting_guidance and getattr(grafting_guidance, "skeleton", None):
        parts.append("")
        parts.append(f"Follow this structure: {grafting_guidance.skeleton.format_for_prompt()}")

    # Structural RAG guidance (rhythm patterns from corpus)
    if structural_guidance:
        parts.append("")
        parts.append(structural_guidance)

    # Constraints. Authors with [DIRECTIVES] get directives true of a real
    # paragraph; the rest keep the old tiers their adapters were trained on.
    directives = _get_directives(adapter_path, worldview)
    if satisfied_by is None and grafting_guidance is not None:
        satisfied_by = getattr(grafting_guidance, "sample_text", None)
    if directives:
        constraints = _build_directive_constraints(directives, satisfied_by)
    elif deterministic_constraints:
        # For testing: include all constraints
        constraints = _format_constraints(
            ALWAYS_CONSTRAINTS + FREQUENT_CONSTRAINTS + [ROTATING_CONSTRAINTS[0]], satisfied_by)
    else:
        constraints = _build_constraints(satisfied_by)
    if constraints:
        parts.append("")
        parts.append(constraints)

    return "\n".join(parts)


def build_persona_prompt(
    content: str,
    structural_guidance: Optional[str] = None,
    grafting_guidance: Optional['GraftingGuidance'] = None,
    target_words: Optional[int] = None,
    deterministic_constraints: bool = False,
    adapter_path: Optional[str] = None,
) -> str:
    """Build a flat prompt in the MLX training format:

    ```
    {persona_frame}

    Write approximately {word_count} words.

    Follow this structure: {skeleton}

    [CONSTRAINT]: Do not use: 'Moreover'...

    {neutral_text}
    ###
    ```

    Chat-template generators should use build_persona_instruction() and pass
    the content separately instead.
    """
    instruction = build_persona_instruction(
        content,
        structural_guidance=structural_guidance,
        grafting_guidance=grafting_guidance,
        target_words=target_words,
        deterministic_constraints=deterministic_constraints,
        adapter_path=adapter_path,
    )
    return f"{instruction}\n\n{content}\n###"
