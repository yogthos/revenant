"""Utility functions for parsing LLM output, especially JSON extraction.

This module provides robust parsing utilities that handle common LLM formatting
errors like single quotes, Markdown code blocks, and extra text.
"""

import json
import re
import ast
from typing import Optional, Union, List, Dict


def extract_json_from_text(text: str) -> Optional[Union[List, Dict]]:
    """Extract and parse JSON from text that may contain formatting errors.

    Handles common LLM output issues:
    - Single quotes instead of double quotes (Python dict format)
    - Markdown code blocks (```json ... ```)
    - Extra text before/after JSON
    - Multiple JSON objects

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON object (list or dict), or None if parsing fails
    """
    if not text or not text.strip():
        return None

    # Step 1: Remove Markdown code blocks if present
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)

    # Step 2: Find JSON array or object pattern
    # Look for first [ or { and matching closing bracket
    array_match = re.search(r'(\[.*\])', text, re.DOTALL)
    dict_match = re.search(r'(\{.*\})', text, re.DOTALL)

    json_str = None
    if array_match:
        json_str = array_match.group(1)
    elif dict_match:
        json_str = dict_match.group(1)

    if not json_str:
        return None

    # Step 3: Fix common LLM formatting errors
    # Replace single quotes around keys with double quotes
    # Pattern: 'key': value -> "key": value
    # Be careful not to replace apostrophes in text values
    # Use a more sophisticated approach: only replace quotes around keys

    # First, try to parse as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Second, try ast.literal_eval (handles Python dict syntax with single quotes)
    # This is safer than aggressive regex replacement
    try:
        return ast.literal_eval(json_str)
    except (ValueError, SyntaxError):
        pass

    # Third, try fixing single quotes around keys
    # Pattern: 'word': -> "word":
    fixed_json = re.sub(r"'(\w+)':", r'"\1":', json_str)

    # Also fix: 'word' : -> "word":
    fixed_json = re.sub(r"'(\w+)'\s*:", r'"\1":', fixed_json)

    try:
        return json.loads(fixed_json)
    except json.JSONDecodeError:
        pass

    # Last resort: try to fix single quotes in string values too
    # This is more dangerous but sometimes necessary
    # Only do this if we're sure it's a JSON structure
    try:
        # Replace all single quotes with double quotes (very aggressive)
        # This might break things, but it's a last resort
        fully_fixed = json_str.replace("'", '"')
        return json.loads(fully_fixed)
    except json.JSONDecodeError:
        pass

    return None

