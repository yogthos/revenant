"""Prompt builder for RAG-based style transfer.

This module constructs highly constrained prompts using RAG data to prevent
'LLM Slop' by explicitly separating vocabulary guidance (situation match)
from structure guidance (structure match).
"""

import random
import re
import textwrap
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from src.analyzer.style_metrics import get_style_vector


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'generator_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


def _analyze_structure(text: str) -> Dict[str, any]:
    """Analyze structural features of a text snippet.

    Args:
        text: Text to analyze.

    Returns:
        Dictionary with structural features:
        - word_count: Total word count
        - sentence_count: Number of sentences
        - avg_sentence_len: Average words per sentence
        - punctuation_pattern: List of punctuation marks used
        - clause_count: Estimated number of clauses
        - voice: "active" or "passive" (heuristic)
        - has_dashes: Whether text contains dashes
        - has_semicolons: Whether text contains semicolons
        - has_parentheses: Whether text contains parentheses
        - has_asterisks: Whether text contains asterisks
        - complexity: "simple", "compound", "complex", or "compound-complex"
    """
    words = text.split()
    word_count = len(words)

    # Count sentences
    sentence_endings = text.count('.') + text.count('!') + text.count('?')
    sentence_count = max(1, sentence_endings)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else word_count

    # Analyze punctuation
    has_dashes = ('—' in text or '-' in text)
    has_semicolons = ';' in text
    has_parentheses = '(' in text and ')' in text
    has_asterisks = '*' in text
    has_commas = ',' in text

    punctuation_pattern = []
    if has_commas:
        punctuation_pattern.append("commas")
    if has_dashes:
        punctuation_pattern.append("dashes")
    if has_semicolons:
        punctuation_pattern.append("semicolons")
    if has_parentheses:
        punctuation_pattern.append("parentheses")
    if has_asterisks:
        punctuation_pattern.append("asterisks")

    # Estimate clause count (rough heuristic: count of conjunctions + relative pronouns)
    clause_indicators = len(re.findall(r'\b(and|or|but|because|since|while|when|if|that|which|who)\b', text, re.IGNORECASE))
    clause_count = max(1, clause_indicators + 1)  # At least 1 clause

    # Heuristic for voice (look for passive indicators)
    passive_indicators = len(re.findall(r'\b(is|are|was|were|been|being)\s+\w+ed\b', text, re.IGNORECASE))
    active_indicators = len(re.findall(r'\b\w+ed\s+(the|a|an|this|that|these|those)\b', text, re.IGNORECASE))
    voice = "passive" if passive_indicators > active_indicators else "active"

    # Determine complexity
    if clause_count == 1:
        complexity = "simple"
    elif clause_count == 2:
        complexity = "compound"
    elif clause_count >= 3:
        complexity = "complex"
    else:
        complexity = "simple"

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_len': avg_sentence_len,
        'punctuation_pattern': punctuation_pattern,
        'clause_count': clause_count,
        'voice': voice,
        'has_dashes': has_dashes,
        'has_semicolons': has_semicolons,
        'has_parentheses': has_parentheses,
        'has_asterisks': has_asterisks,
        'has_commas': has_commas,
        'complexity': complexity
    }


class PromptAssembler:
    """Constructs highly constrained prompts using RAG data to prevent 'LLM Slop'."""

    def __init__(self, target_author_name: str = "Target Author", banned_words: Optional[List[str]] = None):
        """Initialize the prompt assembler.

        Args:
            target_author_name: Name of the target author (for persona definition).
            banned_words: List of words to ban (prevents generic AI language).
        """
        self.author_name = target_author_name

        # Negative constraints to kill "Assistant Voice"
        self.banned_words = banned_words or [
            "delve", "testament", "underscore", "landscape", "tapestry",
            "bustling", "crucial", "meticulous", "comprehensive", "fostering"
        ]

    def build_system_message(self) -> str:
        """Define the rigid persona. The LLM is not a writer; it is a style engine.

        Returns:
            System prompt string.
        """
        template = _load_prompt_template("generator_system.md")
        return template.format(author_name=self.author_name)

    def build_generation_prompt(
        self,
        input_text: str,
        situation_match: Optional[str],
        structure_match: str,
        style_metrics: Optional[Dict[str, float]] = None,
        global_vocab_list: Optional[List[str]] = None
    ) -> str:
        """Assemble the Few-Shot prompt using Dual-RAG data.

        Args:
            input_text: The original input text to rewrite.
            situation_match: Retrieved paragraph for vocabulary grounding (or None).
            structure_match: Retrieved paragraph for rhythm/structure (required).
            style_metrics: Optional style metrics dict (will be extracted from structure_match if not provided).
            global_vocab_list: Optional list of global vocabulary words to inject for variety.

        Returns:
            Complete user prompt string.
        """
        # Extract style metrics from structure_match if not provided
        if style_metrics is None:
            style_vec = get_style_vector(structure_match)
            # Estimate average sentence length from structure_match
            words = structure_match.split()
            sentences = structure_match.count('.') + structure_match.count('!') + structure_match.count('?')
            if sentences > 0:
                avg_sentence_len = len(words) / sentences
            else:
                avg_sentence_len = len(words)
            style_metrics = {'avg_sentence_len': avg_sentence_len}

        # Analyze structure match in detail
        structure_analysis = _analyze_structure(structure_match)

        # Build explicit structure instructions
        structure_instructions = []
        structure_instructions.append(f"- Word count: ~{structure_analysis['word_count']} words")
        structure_instructions.append(f"- Sentence structure: {structure_analysis['complexity']} ({structure_analysis['clause_count']} clauses)")
        structure_instructions.append(f"- Voice: {structure_analysis['voice']}")

        if structure_analysis['punctuation_pattern']:
            structure_instructions.append(f"- Punctuation: Use {', '.join(structure_analysis['punctuation_pattern'])}")

        if structure_analysis['has_dashes']:
            structure_instructions.append("- Include dashes (— or -) for parenthetical or explanatory elements")
        if structure_analysis['has_semicolons']:
            structure_instructions.append("- Use semicolons to connect related independent clauses")
        if structure_analysis['has_parentheses']:
            structure_instructions.append("- Include parenthetical asides using parentheses")
        if structure_analysis['has_asterisks']:
            structure_instructions.append("- Use asterisks (*) for emphasis or special notation")

        structure_instructions_text = "\n".join(structure_instructions)

        # Build situation match content
        if situation_match:
            situation_match_label = "(VOCABULARY PALETTE)"
            situation_match_content = f"""The author has written about this topic before.
Observe their specific word choices and tone in this snippet:
"{situation_match}"

*Instruction: Borrow specific adjectives and verbs from this snippet if they fit.*"""
        else:
            situation_match_label = ""
            situation_match_content = """No direct topic match found in corpus. Rely strictly on the Structural Reference for tone."""

        # Build vocabulary block
        vocab_block = ""
        if global_vocab_list and len(global_vocab_list) > 0:
            sample_size = min(10, len(global_vocab_list))
            flavor_words = ", ".join(random.sample(global_vocab_list, sample_size))
            vocab_block = f"""### VOCABULARY INSPIRATION
1. PRIMARY SOURCE: Use words from the 'Situational Reference' above.
2. SECONDARY SOURCE: Incorporate some of these characteristic author words if they fit:
   [{flavor_words}]

"""

        # Calculate word counts
        input_word_count = len(input_text.split())
        target_word_count = int(input_word_count * 1.2)

        # Load and format the generation prompt template
        template = _load_prompt_template("generation_prompt.md")
        return template.format(
            structure_match=structure_match,
            structure_instructions=structure_instructions_text,
            avg_sentence_len=int(structure_analysis['avg_sentence_len']),
            situation_match_label=situation_match_label,
            situation_match_content=situation_match_content,
            vocab_block=vocab_block,
            input_word_count=input_word_count,
            target_word_count=target_word_count,
            input_text=input_text,
            banned_words=", ".join(self.banned_words)
        )

    def build_blended_prompt(
        self,
        input_text: str,
        bridge_template: str,
        hybrid_vocab: List[str],
        author_a: str,
        author_b: str,
        blend_ratio: float = 0.5
    ) -> str:
        """Build a blended style prompt for mixing two author styles.

        Args:
            input_text: The original input text to rewrite.
            bridge_template: Bridge text that naturally connects the two styles.
            hybrid_vocab: List of words sampled from both authors.
            author_a: First author name.
            author_b: Second author name.
            blend_ratio: Blend ratio (0.0 = All Author A, 1.0 = All Author B).

        Returns:
            Complete user prompt string for blended style generation.
        """
        # Load blended prompt template
        template = _load_prompt_template("generation_blended.md")

        # Format blend description
        if blend_ratio < 0.3:
            blend_desc = f"primarily {author_a} with subtle {author_b} influences"
        elif blend_ratio > 0.7:
            blend_desc = f"primarily {author_b} with subtle {author_a} influences"
        else:
            blend_desc = f"a balanced blend of {author_a} and {author_b}"

        # Format vocabulary list
        vocab_text = ", ".join(hybrid_vocab) if hybrid_vocab else "N/A"

        return template.format(
            bridge_template=bridge_template,
            hybrid_vocab=vocab_text,
            author_a=author_a,
            author_b=author_b,
            blend_desc=blend_desc,
            input_text=input_text
        )

