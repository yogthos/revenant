"""Build persona-injected prompts for subjective style transfer.

The key insight: instead of melodrama, we use "Dense Monograph" style.
This mimics academic human writing patterns that AI detectors interpret as "Real":
- Syntactic complexity (garden-path sentences, relative clauses)
- Nominalization (abstract nouns over active verbs)
- Clinical detachment over emotional description
"""

from typing import List, Optional, TYPE_CHECKING
from .config import PersonaConfig

if TYPE_CHECKING:
    from ..rag.structural_grafter import GraftingGuidance


def build_persona_prompt(
    content: str,
    author: str,
    persona: PersonaConfig,
    vocabulary_palette: Optional[List[str]] = None,
    structural_guidance: Optional[str] = None,
    grafting_guidance: Optional['GraftingGuidance'] = None,
) -> str:
    """Build a prompt that forces dense, academic output.

    Args:
        content: The neutralized content to transform.
        author: Author name.
        persona: Persona configuration for this author.
        vocabulary_palette: Optional list of author-characteristic words/phrases.
        structural_guidance: Optional structural patterns from RAG.
        grafting_guidance: Optional grafting guidance with sample + skeleton.

    Returns:
        Formatted prompt string.
    """
    # Build vocabulary section if provided
    vocab_section = ""
    if vocabulary_palette:
        vocab_items = ", ".join(vocabulary_palette[:8])
        vocab_section = f"\n[VOCABULARY PALETTE]: {vocab_items}\n"

    # Build structural guidance section if provided
    structure_section = ""
    if structural_guidance:
        structure_section = f"\n[STRUCTURAL PATTERNS]:\n{structural_guidance}\n"

    # Build grafting section if provided (takes precedence over structural guidance)
    grafting_section = ""
    if grafting_guidance:
        grafting_section = f"\n{grafting_guidance.format_for_prompt()}\n"

    prompt = f"""You are {author}.

[CRITICAL CONSTRAINT]: The output must pass AI detection by mimicking "Human Academic Density."

[STYLE RULES]:
1. DENSITY OVER CLARITY: Complicate simple actions into complex processes.
   - Bad (AI): "The bank failed."
   - Good: "The institution, having long teetered upon a foundation of doubtful solvency, underwent a sudden and ruinous dissolution."

2. SYNTACTIC INTERRUPTION: Interrupt main clauses with context.
   - Use em-dashes (—) to insert modifying clauses mid-sentence.
   - Use "which" and "wherein" clauses to extend sentences.
   - Example: "The report—though initially dismissed as conjecture—contains truths that..."

3. GARDEN-PATH SENTENCES: Separate subject from verb with long modifiers.
   - Example: "The phenomenon, which had been observed across multiple centuries in disparate cultures, defies..."

4. NOMINALIZATION: Prefer abstract nouns over active verbs.
   - Bad: "We found the error."
   - Good: "The error was discovered to lie within the fundamental architecture..."

5. PARENTHETICAL ASIDES: Insert doubts, qualifications, references.
   - Example: "...a fact which I initially doubted, though later evidence proved otherwise..."

6. NO MELODRAMA: Horror comes from implication, not adjectives.
   - Avoid: "spooky," "scary," "terrifying"
   - Instead: Describe structure and process with clinical detachment.

7. PRESERVE ALL CONTENT: Every fact from input MUST appear in output.
{vocab_section}{structure_section}{grafting_section}
Rewrite as {author}'s academic/scientific voice while preserving ALL content:

{content}
###
"""

    return prompt
