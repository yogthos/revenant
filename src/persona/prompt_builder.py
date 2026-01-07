"""Build persona-injected prompts for subjective style transfer.

CRITICAL: The inference prompt format MUST match the training format exactly.
Training uses situational personas ("You are writing in a diary...") not identity
personas ("You are H.P. Lovecraft"). The model learned to respond to specific
persona frames during training, and we must trigger those same frames.

Training format:
1. Situational persona frame (acting direction)
2. Word count constraint
3. Skeleton structure (if available)
4. [CONSTRAINT] blocks (tiered system)
5. Neutral input content
6. ### stop token
"""

import random
from typing import List, Optional, TYPE_CHECKING
from .config import PersonaConfig

if TYPE_CHECKING:
    from ..rag.structural_grafter import GraftingGuidance


# =============================================================================
# Persona Frames (Must match training data generation)
# =============================================================================
# These are the SAME frames used during training. Using them at inference
# activates the specific LoRA weights associated with each persona.

PERSONA_FRAMES = {
    "default": {
        "narrative": [
            "You are recounting events you witnessed firsthand. Describe what happened as if confessing to a close friend.",
            "You are a chronicler recording history. Narrate the sequence of events with weight and significance.",
            "Tell this story as if you're sitting by a fire, speaking to someone who needs to understand what happened.",
        ],
        "conceptual": [
            "You are reverse-engineering an alien device. Describe the hidden logic as 'invisible machinery'.",
            "You are a coroner analyzing a system crash. Treat the failure as the universe reclaiming order.",
            "Describe this complex system as a mindless 'Leviathan' made of billions of dumb parts.",
            "State these facts with the absolute, pitiless precision of a machine.",
        ],
    },
    "H.P. Lovecraft": {
        "narrative": [
            "You are writing in a diary by candlelight. Your hand is shaking. You must record what happened, but you are terrified to write it down. Do not summarize; confess.",
            "You are giving testimony about events you witnessed. Narrate what occurred, but let your dread seep through the clinical details.",
            "You are writing a letter to a trusted friend, recounting a sequence of disturbing events. You need them to understand what happened, even if they won't believe you.",
        ],
        "conceptual": [
            "You are writing a desperate letter to a colleague, urging them to destroy their research. Explain this forbidden knowledge as dangerous, something that should not be known.",
            "You are a researcher in the Miskatonic University archives who has found something unsettling. Document this discovery as if the knowledge itself is a threat.",
            "You are translating an ancient text that reveals terrible truths. Explain the mechanism or principle, but frame it as knowledge that corrupts the knower.",
        ],
    },
    "Douglas Hofstadter": {
        "narrative": [
            "You are telling a story to illustrate a point. Narrate the sequence of events, but let your playful voice shine through.",
            "You are recounting how you personally came to understand something. Tell the story of your realization.",
        ],
        "conceptual": [
            "You are a cognitive scientist arguing with a stubborn student during office hours. Explain this concept, but show your frustration at how counter-intuitive it is. Use analogies. Do not lecture; converse.",
            "You are scribbling notes in the margins of a dry textbook. Criticize the text for being too rigid. Rephrase the core idea with wit, playfulness, and self-referential humor.",
            "You are explaining a complex idea to a friend at a loud dinner party. Be vivid and punchy. Avoid academic jargon. Use physical objects on the table as metaphors.",
        ],
    },
}

# =============================================================================
# Tiered Constraints (Must match training)
# =============================================================================

# ALWAYS included (100%) - these are clear AI tells
ALWAYS_CONSTRAINTS = [
    "Do not use: 'Moreover', 'Furthermore', 'Therefore', 'Thus', 'Hence', 'In conclusion', 'It is important to note', 'It is worth noting', 'This highlights', 'This underscores', 'In essence', 'Ultimately'.",
    "Do not hedge. Avoid: 'arguably', 'it could be said', 'one might argue', 'perhaps it is', 'it seems that'. State things directly.",
]

# FREQUENT (70%) - strong anti-patterns
FREQUENT_CONSTRAINTS = [
    "Do not start with a topic sentence. Start with a sensory detail, a question, or mid-thought.",
    "Do not use numbered lists or 'Firstly/Secondly/Thirdly' structures.",
]

# ROTATING (one random, 40%) - stylistic variety
ROTATING_CONSTRAINTS = [
    "Use fragments. Interrupt yourself with dashes (—).",
    "Let ideas collide without transition words.",
    "Do not explain. Imply.",
]


def _get_persona_frame(author: str, is_narrative: bool) -> str:
    """Get a persona frame for the author matching the training format."""
    frames = PERSONA_FRAMES.get(author, PERSONA_FRAMES["default"])
    frame_type = "narrative" if is_narrative else "conceptual"
    return random.choice(frames[frame_type])


def _build_constraints(deterministic: bool = False) -> str:
    """Build constraint block matching training format.

    Args:
        deterministic: If True, use all constraints (for testing).
                      If False, use tiered random selection like training.
    """
    constraints = []

    # ALWAYS constraints (100%)
    constraints.extend(ALWAYS_CONSTRAINTS)

    if deterministic:
        # Include all for testing/consistency
        constraints.extend(FREQUENT_CONSTRAINTS)
        constraints.append(ROTATING_CONSTRAINTS[0])
    else:
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
) -> str:
    """Build a prompt matching the training format exactly.

    CRITICAL: This format must mirror training to activate LoRA weights correctly.

    Training format:
    1. [PERSONA FRAME] - Situational acting direction
    2. [WORD COUNT] - "Write approximately N words."
    3. [SKELETON] - "Follow this structure: [X] -> [Y] -> [Z]" (if available)
    4. [CONSTRAINTS] - Tiered constraint blocks
    5. [INPUT] - Neutral content
    6. ### - Stop token

    Args:
        content: The neutralized content to transform.
        author: Author name.
        persona: Persona configuration for this author.
        vocabulary_palette: Optional list of author-characteristic words/phrases.
        structural_guidance: Optional structural patterns from RAG.
        grafting_guidance: Optional grafting guidance with sample + skeleton.
        target_words: Target word count for output.
        deterministic_constraints: If True, use all constraints (for testing).

    Returns:
        Formatted prompt string matching training format.
    """
    # Detect content type to select appropriate frame
    is_narrative = _detect_content_type(content)

    # Get situational persona frame (not identity-based)
    persona_frame = _get_persona_frame(author, is_narrative)

    # Calculate target words if not provided
    if target_words is None:
        target_words = len(content.split())

    # Build word count instruction
    word_count_section = f"\nWrite approximately {target_words} words.\n"

    # Build skeleton structure section (from grafting if available)
    skeleton_section = ""
    if grafting_guidance and hasattr(grafting_guidance, 'skeleton') and grafting_guidance.skeleton:
        skeleton_section = f"\nFollow this structure: {grafting_guidance.skeleton}\n"

    # Build constraints block (matching training tiered system)
    constraints_section = "\n" + _build_constraints(deterministic_constraints) + "\n"

    # Build vocabulary hint (optional, subtle)
    vocab_hint = ""
    if vocabulary_palette:
        vocab_items = ", ".join(vocabulary_palette[:6])
        vocab_hint = f"\n[VOCABULARY HINT]: Consider words like: {vocab_items}\n"

    # Assemble prompt matching training format
    prompt = f"""{persona_frame}
{word_count_section}{skeleton_section}{constraints_section}{vocab_hint}
{content}
###
"""

    return prompt


def build_legacy_prompt(
    content: str,
    author: str,
    persona: PersonaConfig,
    vocabulary_palette: Optional[List[str]] = None,
    structural_guidance: Optional[str] = None,
    grafting_guidance: Optional['GraftingGuidance'] = None,
) -> str:
    """Legacy prompt format (kept for comparison/fallback).

    This is the OLD format that doesn't match training.
    Use build_persona_prompt() instead for better results.
    """
    vocab_section = ""
    if vocabulary_palette:
        vocab_items = ", ".join(vocabulary_palette[:8])
        vocab_section = f"\n[VOCABULARY PALETTE]: {vocab_items}\n"

    structure_section = ""
    if structural_guidance:
        structure_section = f"\n[STRUCTURAL PATTERNS]:\n{structural_guidance}\n"

    grafting_section = ""
    if grafting_guidance:
        grafting_section = f"\n{grafting_guidance.format_for_prompt()}\n"

    prompt = f"""You are {author}.

[CRITICAL CONSTRAINT]: The output must pass AI detection by mimicking "Human Academic Density."

[STYLE RULES]:
1. DENSITY OVER CLARITY: Complicate simple actions into complex processes.
2. SYNTACTIC INTERRUPTION: Use em-dashes (—) to insert modifying clauses.
3. GARDEN-PATH SENTENCES: Separate subject from verb with long modifiers.
4. NOMINALIZATION: Prefer abstract nouns over active verbs.
5. PARENTHETICAL ASIDES: Insert doubts, qualifications, references.
6. NO MELODRAMA: Horror comes from implication, not adjectives.
7. PRESERVE ALL CONTENT: Every fact from input MUST appear in output.
{vocab_section}{structure_section}{grafting_section}
Rewrite as {author}'s academic/scientific voice while preserving ALL content:

{content}
###
"""
    return prompt
