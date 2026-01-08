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

# =============================================================================
# Author Worldview - Loaded from prompts/{filename} specified in config.json
# =============================================================================
# The worldview isn't just a writing style - it's a LENS through which the author
# perceives reality. The model must THINK like the author, not just write like them.
#
# Configure in config.json under lora.worldview (filename in prompts/ folder)

from pathlib import Path

def _load_worldview() -> str:
    """Load author worldview from file specified in config.json lora.worldview."""
    try:
        from ..config import load_config
        config = load_config()
        if config.lora.worldview:
            prompts_dir = Path(__file__).parent.parent.parent / "prompts"
            worldview_file = prompts_dir / config.lora.worldview
            if worldview_file.exists():
                return worldview_file.read_text(encoding="utf-8")
    except Exception:
        pass
    # Fallback
    return "You are a distinctive author. Write with your unique voice and perspective."


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

# POSITIVE STYLE GUIDANCE (100%) - What human authors DO
POSITIVE_STYLE_GUIDANCE = [
    "ENSURE GRAMMATICAL CORRECTNESS: Every sentence must be grammatically sound. Check that subjects and verbs agree, and that each sentence parses correctly as English.",
    "ENSURE COHERENCE: Each sentence must logically follow from the previous. Use pronouns ('this', 'it', 'such'), synonyms, or parallel structure to connect ideas naturally.",
    "EMBED your emotional reaction in the prose: 'horrible to relate', 'I confess with some reluctance', 'what words can capture'.",
    "INTERRUPT yourself with em-dashes for sudden asides—like this—revealing your inner turmoil.",
    "VARY sentence length: mix short declarative sentences with longer, more complex ones.",
]

# ANTI-AI-TELL CONSTRAINTS (100%) - Critical patterns that reveal AI authorship
ANTI_AI_TELL_CONSTRAINTS = [
    "AVOID generic topic sentences like 'This demonstrates...', 'It is clear that...', 'This is important because...'. Instead, START with a concrete observation, specific fact, or direct statement.",
    "NEVER use balanced 'A, B, and C' lists. Break into separate sentences or asymmetric phrasing.",
    "LIMIT subordinate clauses: one 'which' or 'that' per sentence maximum.",
    "END decisively. No trailing '...and which could be' or '...or become' clauses.",
]

# FREQUENT (70%) - strong anti-patterns
FREQUENT_CONSTRAINTS = [
    "AVOID dry topic sentences. Open with a specific observation, a physical detail, or a direct claim—but ensure the sentence is grammatically complete.",
    "Do not use numbered lists or 'Firstly/Secondly/Thirdly' structures.",
]

# ROTATING (one random, 40%) - stylistic variety
ROTATING_CONSTRAINTS = [
    "Use fragments. Interrupt yourself with dashes (—).",
    "Let ideas collide without transition words.",
    "Do not explain. Imply.",
]

# =============================================================================
# Stylistic Hints (Add variety across paragraphs)
# =============================================================================

# PUNCTUATION_HINTS (30%) - vary punctuation emphasis
PUNCTUATION_HINTS = [
    "Use semicolons to connect related clauses; let thoughts flow together.",
    "Interrupt yourself with em-dashes—asides that reveal inner thought—mid-sentence.",
    "Use parentheticals (qualifications, doubts, afterthoughts) to add texture.",
    "Deploy colons to introduce: revelations, lists, or explanations.",
    "Let ellipses trail off into implication...",
]

# RHYTHM_HINTS (30%) - vary sentence rhythm emphasis
RHYTHM_HINTS = [
    "Vary sentence length dramatically. Short punch. Then sprawling complexity that winds through multiple clauses before finally arriving at its destination.",
    "Build momentum: short sentence, longer sentence, longest sentence with cascading clauses.",
    "Alternate between staccato declarations and flowing elaborations.",
    "Use repetition for emphasis. Use repetition for rhythm. Use repetition.",
    "Let one long sentence do the work of three, connected by semicolons and conjunctions.",
]

# OPENING_HINTS (30%) - vary how paragraphs begin
OPENING_HINTS = [
    "Begin with a concrete image or physical sensation.",
    "Start mid-action, as if continuing a thought already begun.",
    "Open with a question that the paragraph will circle around.",
    "Begin with a date, time, or specific number.",
    "Start with 'I' or 'We' for immediacy and personal witness.",
    "Open with a subordinate clause: 'Although...', 'When...', 'Before...'",
]

# EMOTIONAL_REGISTERS (25%) - vary emotional framing
EMOTIONAL_REGISTERS = {
    "H.P. Lovecraft": [
        "Write with mounting unease—each sentence should increase the dread.",
        "Maintain clinical detachment even as horror accumulates.",
        "Let reluctance seep through: you don't want to write this, but you must.",
        "Convey the weight of forbidden knowledge pressing on the narrator.",
    ],
    "Douglas Hofstadter": [
        "Let playful curiosity bubble through the explanation.",
        "Show visible frustration at the difficulty of conveying the idea.",
        "Write with the joy of someone sharing a favorite puzzle.",
        "Be self-deprecating about the inadequacy of language.",
    ],
    "default": [
        "Write with conviction, as if these truths must be recorded.",
        "Let personal investment show through professional distance.",
        "Convey the struggle to articulate something important.",
        "Write as if confessing something you've been reluctant to admit.",
    ],
}


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

    # POSITIVE guidance FIRST (100%) - more effective than negatives
    constraints.extend(POSITIVE_STYLE_GUIDANCE)

    # ALWAYS constraints (100%)
    constraints.extend(ALWAYS_CONSTRAINTS)

    # ANTI-AI-TELL constraints (100%) - Critical for avoiding AI detection
    constraints.extend(ANTI_AI_TELL_CONSTRAINTS)

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


def _build_stylistic_hints(author: str, deterministic: bool = False) -> str:
    """Build stylistic hints section to add variety across paragraphs.

    These hints are probabilistically selected to ensure each paragraph
    gets slightly different guidance, preventing monotonous output.

    Args:
        author: Author name for author-specific emotional registers.
        deterministic: If True, include one of each type (for testing).
    """
    hints = []

    if deterministic:
        # Include one of each for testing
        hints.append(PUNCTUATION_HINTS[0])
        hints.append(RHYTHM_HINTS[0])
        hints.append(OPENING_HINTS[0])
        registers = EMOTIONAL_REGISTERS.get(author, EMOTIONAL_REGISTERS["default"])
        hints.append(registers[0])
    else:
        # Probabilistic selection for variety

        # Punctuation hint (30%)
        if random.random() < 0.30:
            hints.append(random.choice(PUNCTUATION_HINTS))

        # Rhythm hint (30%)
        if random.random() < 0.30:
            hints.append(random.choice(RHYTHM_HINTS))

        # Opening hint (30%)
        if random.random() < 0.30:
            hints.append(random.choice(OPENING_HINTS))

        # Emotional register (25%) - author-specific
        if random.random() < 0.25:
            registers = EMOTIONAL_REGISTERS.get(author, EMOTIONAL_REGISTERS["default"])
            hints.append(random.choice(registers))

    if not hints:
        return ""

    return "\n[STYLE HINT]: " + "\n[STYLE HINT]: ".join(hints)


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

    # Get author worldview from config - this is CRITICAL for embodiment
    author_worldview = _load_worldview()

    # Build the transformation directive - this is the most important instruction
    transformation_directive = f"""
=== CRITICAL: YOU ARE THIS AUTHOR ===
{author_worldview}

=== TRANSFORMATION DIRECTIVE ===
Do NOT paraphrase. Do NOT just swap synonyms. TRANSFORM.

You must REIMAGINE this content through your eyes. Ask yourself:
- How would I (the author) PERCEIVE this subject?
- What emotions does it stir in me?
- What metaphors and imagery come naturally to my mind?
- How does this connect to my deepest concerns and obsessions?

Write approximately {target_words} words. You may write MORE if your voice demands it.
The content below is raw material. Digest it. Then EXPRESS IT AS ONLY YOU CAN.
"""

    # Build skeleton structure section (from grafting if available)
    skeleton_section = ""
    if grafting_guidance and hasattr(grafting_guidance, 'skeleton') and grafting_guidance.skeleton:
        skeleton_section = f"\nFollow this structure: {grafting_guidance.skeleton}\n"

    # Build structural RAG guidance section (rhythm, vocabulary, transitions, organic complexity)
    # This is the CRITICAL section that provides author-specific patterns from ChromaDB
    structural_section = ""
    if structural_guidance:
        structural_section = f"""
[CRITICAL - AUTHOR STYLE PATTERNS - MANDATORY]:
These patterns are extracted from the author's actual writing. Follow them precisely to achieve authentic style:

{structural_guidance}

[END AUTHOR PATTERNS]
"""

    # Build constraints block (matching training tiered system)
    constraints_section = "\n" + _build_constraints(deterministic_constraints) + "\n"

    # Build stylistic hints (probabilistically selected for variety)
    stylistic_hints = _build_stylistic_hints(author, deterministic_constraints)

    # Build vocabulary hint (optional, subtle)
    vocab_hint = ""
    if vocabulary_palette:
        vocab_items = ", ".join(vocabulary_palette[:6])
        vocab_hint = f"\n[VOCABULARY HINT]: Consider words like: {vocab_items}\n"

    # Assemble prompt with author embodiment at the core
    # Order: transformation directive → persona frame → skeleton → structural RAG → constraints → style hints → vocab → content
    prompt = f"""{transformation_directive}

{persona_frame}
{skeleton_section}{structural_section}{constraints_section}{stylistic_hints}{vocab_hint}
{content}
###
"""

    return prompt


