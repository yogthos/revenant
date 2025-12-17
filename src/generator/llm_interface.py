"""LLM interface for constrained sentence generation.

This module handles communication with LLM APIs (DeepSeek) to generate
sentences that preserve semantic meaning while following specific
structural templates and vocabulary requirements.
"""

import json
import os
import re
from typing import Dict, List, Optional
import requests

from src.models import ContentUnit
from src.generator.prompt_builder import PromptAssembler
from src.analyzer.style_metrics import get_style_vector


def _load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from config.json.

    Args:
        config_path: Path to the config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def _call_deepseek_api(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    api_url: str,
    model: str = "deepseek-chat"
) -> str:
    """Call DeepSeek API to generate text.

    Args:
        system_prompt: System prompt for the LLM.
        user_prompt: User prompt with the request.
        api_key: DeepSeek API key.
        api_url: DeepSeek API URL.
        model: Model name to use.

    Returns:
        Generated text response.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  # Very low temperature for precise structure matching
        "max_tokens": 200
    }

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError(f"Unexpected API response: {result}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API request failed: {e}")


def generate_sentence(
    content_unit: ContentUnit,
    structure_match: str,
    situation_match: Optional[str] = None,
    config_path: str = "config.json",
    hint: Optional[str] = None,
    target_author_name: str = "Target Author",
    global_vocab_list: Optional[List[str]] = None,
    author_names: Optional[List[str]] = None,
    blend_ratio: Optional[float] = None,
    use_fallback_structure: bool = False
) -> str:
    """Generate a sentence using LLM with dual RAG references.

    Generates a sentence that:
    - Preserves the EXACT semantic meaning from content_unit
    - Matches the sentence structure and rhythm of structure_match
    - Uses vocabulary tone from situation_match (if available)

    Args:
        content_unit: ContentUnit containing SVO triples, entities, and original text.
        structure_match: Reference paragraph for rhythm/structure (required).
        situation_match: Reference paragraph for vocabulary grounding (optional).
        config_path: Path to configuration file.
        hint: Optional hint/feedback from previous attempt to improve generation.
        target_author_name: Name of target author for persona (default: "Target Author").
        global_vocab_list: Optional list of global vocabulary words to inject for variety.

    Returns:
        Generated sentence string.
    """
    # Load configuration
    config = _load_config(config_path)
    provider = config.get("provider", "deepseek")

    if provider == "deepseek":
        deepseek_config = config.get("deepseek", {})
        api_key = deepseek_config.get("api_key")
        api_url = deepseek_config.get("api_url")
        model = deepseek_config.get("editor_model", "deepseek-chat")

        if not api_key or not api_url:
            raise ValueError("DeepSeek API key or URL not found in config")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Initialize prompt assembler
    assembler = PromptAssembler(target_author_name=target_author_name)

    # Build system prompt using PromptAssembler
    system_prompt = assembler.build_system_message()

    # Add examples to system prompt
    from pathlib import Path
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    examples_path = prompts_dir / "generator_examples.md"
    if examples_path.exists():
        examples = examples_path.read_text().strip()
        system_prompt += "\n\n" + examples

    # Extract style metrics from structure_match
    style_vec = get_style_vector(structure_match)
    words = structure_match.split()
    sentences = structure_match.count('.') + structure_match.count('!') + structure_match.count('?')
    if sentences > 0:
        avg_sentence_len = len(words) / sentences
    else:
        avg_sentence_len = len(words)

    style_metrics = {'avg_sentence_len': avg_sentence_len}

    # Build user prompt using PromptAssembler (blend mode or single-author mode)
    if author_names and len(author_names) >= 2 and blend_ratio is not None:
        # Blend mode: use blended prompt
        user_prompt = assembler.build_blended_prompt(
            input_text=content_unit.original_text,
            bridge_template=structure_match,
            hybrid_vocab=global_vocab_list or [],
            author_a=author_names[0],
            author_b=author_names[1],
            blend_ratio=blend_ratio
        )
    else:
        # Single-author mode: use regular prompt
        user_prompt = assembler.build_generation_prompt(
            input_text=content_unit.original_text,
            situation_match=situation_match,
            structure_match=structure_match,
            style_metrics=style_metrics,
            global_vocab_list=global_vocab_list,
            use_fallback_structure=use_fallback_structure
        )

    # Add entity preservation if needed
    if content_unit.entities:
        user_prompt += "\n\n"
        user_prompt += f"IMPORTANT: Preserve these entities exactly (DO NOT add others): {', '.join(content_unit.entities)}"

    # Add content words hint if needed
    if content_unit.content_words:
        important_words = content_unit.content_words[:20]
        user_prompt += "\n"
        user_prompt += f"Key concepts to include: {', '.join(important_words)}"

    # Add hint from previous attempt if provided (for retries)
    # Use Chain-of-Thought format for retries
    if hint:
        user_prompt += "\n\n"
        user_prompt += "--- RETRY MODE: CHAIN-OF-THOUGHT CORRECTION ---\n\n"
        user_prompt += f"CRITIC FEEDBACK: {hint}\n\n"
        user_prompt += "TASK:\n"
        user_prompt += "1. Analyze WHY the previous attempt failed the critic's specific rule.\n"
        user_prompt += "2. Write a 'Plan of Correction' (1 sentence).\n"
        user_prompt += "3. Generate the final corrected text.\n\n"
        user_prompt += "Output format:\n"
        user_prompt += "PLAN: [Your reasoning]\n"
        user_prompt += "TEXT: [The corrected text]"
        # If hint contains length information, emphasize it
        if "words" in hint.lower() and ("delete" in hint.lower() or "expand" in hint.lower() or "add" in hint.lower()):
            user_prompt += "\n\nCRITICAL: The length constraint above is a hard requirement. You MUST follow the exact word count instruction."

    # Call API
    generated_text = _call_deepseek_api(system_prompt, user_prompt, api_key, api_url, model)

    # Parse CoT response if hint was provided (retry mode)
    if hint:
        # Try to extract text after "TEXT:" marker
        text_match = re.search(r'TEXT:\s*(.+?)(?:\n\n|$)', generated_text, re.DOTALL | re.IGNORECASE)
        if text_match:
            generated_text = text_match.group(1).strip()
        # Fallback: if no TEXT marker, try to find the last paragraph or sentence
        elif "PLAN:" in generated_text.upper():
            # Split by PLAN: and take everything after the last occurrence
            parts = re.split(r'PLAN:', generated_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Take the last part and try to extract text
                last_part = parts[-1]
                # Remove "TEXT:" if present and take the rest
                text_cleaned = re.sub(r'^TEXT:\s*', '', last_part, flags=re.IGNORECASE).strip()
                if text_cleaned:
                    generated_text = text_cleaned

    return generated_text

