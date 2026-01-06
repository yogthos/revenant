"""Track and preserve footnote references during style transfer.

References like [^1], [^2] are extracted before transfer and reinjected
after, using entity matching to find the correct attachment point.

Example:
    Input: "Einstein[^1] developed relativity[^2] in 1905."
    After RTT: "Einstein developed relativity in 1905."
    After LoRA: "The physicist Einstein formulated his theory of relativity in 1905."
    After reinject: "The physicist Einstein[^1] formulated his theory of relativity[^2] in 1905."
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Pattern for footnote references: [^1], [^2], [^note], etc.
REFERENCE_PATTERN = re.compile(r'\[\^([^\]]+)\]')

# Pattern for inline citations: [1], [2], [1,2], [1-3], etc.
CITATION_PATTERN = re.compile(r'\[(\d+(?:[,\-]\d+)*)\]')


@dataclass
class Reference:
    """A tracked reference marker."""
    marker: str           # The full marker, e.g., "[^1]"
    ref_id: str          # The reference ID, e.g., "1"
    attached_to: str     # The word/phrase it's attached to
    context: str         # Surrounding context for matching
    position: int        # Character position in original text


@dataclass
class ReferenceMap:
    """Map of references extracted from source text."""
    references: List[Reference] = field(default_factory=list)
    # Map from attached entity -> list of references
    entity_refs: Dict[str, List[Reference]] = field(default_factory=dict)

    def add(self, ref: Reference) -> None:
        """Add a reference to the map."""
        self.references.append(ref)

        # Index by attached entity (lowercased for matching)
        key = ref.attached_to.lower().strip()
        if key not in self.entity_refs:
            self.entity_refs[key] = []
        self.entity_refs[key].append(ref)

    def get_refs_for_entity(self, entity: str) -> List[Reference]:
        """Get all references attached to an entity."""
        key = entity.lower().strip()
        return self.entity_refs.get(key, [])

    def has_references(self) -> bool:
        """Check if any references were found."""
        return len(self.references) > 0


def extract_references(text: str) -> Tuple[str, ReferenceMap]:
    """Extract all references from text and return cleaned text + reference map.

    Args:
        text: Source text with references.

    Returns:
        Tuple of (text_without_references, reference_map)
    """
    ref_map = ReferenceMap()

    # Find all footnote references [^N]
    for match in REFERENCE_PATTERN.finditer(text):
        marker = match.group(0)  # Full match: [^1]
        ref_id = match.group(1)  # Just the ID: 1
        position = match.start()

        # Find what the reference is attached to (word before the reference)
        prefix = text[:position].rstrip()
        attached_to = _extract_attached_word(prefix)

        # Get surrounding context for better matching
        context_start = max(0, position - 50)
        context_end = min(len(text), match.end() + 50)
        context = text[context_start:context_end]

        ref = Reference(
            marker=marker,
            ref_id=ref_id,
            attached_to=attached_to,
            context=context,
            position=position,
        )
        ref_map.add(ref)
        logger.debug(f"Found reference {marker} attached to '{attached_to}'")

    # Find inline citations [N] - but be careful not to match other brackets
    for match in CITATION_PATTERN.finditer(text):
        marker = match.group(0)
        ref_id = match.group(1)
        position = match.start()

        # Skip if this looks like it's part of markdown or code
        if position > 0 and text[position-1] in '![':
            continue

        prefix = text[:position].rstrip()
        attached_to = _extract_attached_word(prefix)

        context_start = max(0, position - 50)
        context_end = min(len(text), match.end() + 50)
        context = text[context_start:context_end]

        ref = Reference(
            marker=marker,
            ref_id=ref_id,
            attached_to=attached_to,
            context=context,
            position=position,
        )
        ref_map.add(ref)
        logger.debug(f"Found citation {marker} attached to '{attached_to}'")

    # Remove references from text for processing
    cleaned = REFERENCE_PATTERN.sub('', text)
    cleaned = CITATION_PATTERN.sub('', cleaned)
    # Clean up any double spaces left behind
    cleaned = re.sub(r'  +', ' ', cleaned)

    if ref_map.has_references():
        logger.info(f"Extracted {len(ref_map.references)} references from text")

    return cleaned, ref_map


def _extract_attached_word(prefix: str) -> str:
    """Extract the word/phrase that a reference is attached to.

    The reference is typically attached to:
    1. The immediately preceding word
    2. A quoted phrase ending before the reference
    3. A proper noun phrase

    Args:
        prefix: Text before the reference marker.

    Returns:
        The word or phrase the reference is attached to.
    """
    if not prefix:
        return ""

    # Check for quoted text ending at reference
    if prefix.endswith('"') or prefix.endswith("'"):
        # Find matching opening quote
        quote_char = prefix[-1]
        quote_start = prefix.rfind(quote_char, 0, len(prefix)-1)
        if quote_start != -1:
            return prefix[quote_start+1:-1]

    # Check for parenthetical ending at reference
    if prefix.endswith(')'):
        paren_start = prefix.rfind('(')
        if paren_start != -1:
            return prefix[paren_start+1:-1]

    # Default: extract the last word(s)
    # Handle multi-word proper nouns (e.g., "Karl Marx")
    words = prefix.split()
    if not words:
        return ""

    last_word = words[-1].strip('.,;:!?')

    # Check if this might be part of a proper noun phrase
    # (previous word is also capitalized)
    if len(words) >= 2:
        prev_word = words[-2].strip('.,;:!?')
        if prev_word and prev_word[0].isupper() and last_word and last_word[0].isupper():
            return f"{prev_word} {last_word}"

    return last_word


def reinject_references(
    output: str,
    ref_map: ReferenceMap,
    source_entities: Optional[List[str]] = None,
) -> str:
    """Reinject references into styled output based on entity matching.

    Args:
        output: Styled text without references.
        ref_map: Reference map from extraction.
        source_entities: Optional list of entities from source graph.

    Returns:
        Output text with references reinjected.
    """
    if not ref_map.has_references():
        return output

    result = output
    injected_count = 0

    # Process each reference
    for ref in ref_map.references:
        attached = ref.attached_to
        if not attached:
            continue

        # Try to find the attached word/phrase in output
        injection_point = _find_injection_point(result, attached, ref.context)

        if injection_point is not None:
            # Inject the reference marker after the matched word
            result = result[:injection_point] + ref.marker + result[injection_point:]
            injected_count += 1
            logger.debug(f"Injected {ref.marker} after '{attached}'")
        else:
            # Try fuzzy matching with similar words
            fuzzy_point = _find_fuzzy_injection_point(result, attached)
            if fuzzy_point is not None:
                result = result[:fuzzy_point] + ref.marker + result[fuzzy_point:]
                injected_count += 1
                logger.debug(f"Fuzzy injected {ref.marker} near '{attached}'")
            else:
                logger.warning(f"Could not find injection point for {ref.marker} (attached to '{attached}')")

    if injected_count > 0:
        logger.info(f"Reinjected {injected_count}/{len(ref_map.references)} references")

    return result


def _find_injection_point(text: str, attached: str, context: str) -> Optional[int]:
    """Find where to inject a reference in the output text.

    Args:
        text: Output text to search.
        attached: Word/phrase the reference was attached to.
        context: Original context around the reference.

    Returns:
        Character position to inject at, or None if not found.
    """
    if not attached:
        return None

    # Try exact match first (case-insensitive)
    pattern = re.compile(
        r'\b' + re.escape(attached) + r'\b',
        re.IGNORECASE
    )

    matches = list(pattern.finditer(text))

    if not matches:
        return None

    if len(matches) == 1:
        # Only one match - use it
        return matches[0].end()

    # Multiple matches - try to disambiguate using context
    # Look for context words that appear near the match
    context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))

    best_match = None
    best_score = 0

    for match in matches:
        # Get surrounding context in output
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        local_context = text[start:end].lower()

        # Score by number of context words present
        score = sum(1 for word in context_words if word in local_context)

        if score > best_score:
            best_score = score
            best_match = match

    if best_match:
        return best_match.end()

    # Fall back to first match
    return matches[0].end()


def _find_fuzzy_injection_point(text: str, attached: str) -> Optional[int]:
    """Find injection point using fuzzy matching.

    Handles cases where the word was transformed during style transfer
    (e.g., "Einstein" -> "the physicist Einstein").

    Args:
        text: Output text to search.
        attached: Original attached word.

    Returns:
        Character position to inject at, or None if not found.
    """
    if not attached or len(attached) < 3:
        return None

    # Try to find the word as part of a larger phrase
    # This handles "Einstein" -> "Einstein's theory" -> "the physicist Einstein"

    # First try: word at end of a phrase (common for names)
    pattern = re.compile(
        r'\b[\w\s]{0,30}' + re.escape(attached) + r'\b',
        re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        # Find where 'attached' ends within the match
        attached_match = re.search(
            r'\b' + re.escape(attached) + r'\b',
            match.group(0),
            re.IGNORECASE
        )
        if attached_match:
            return match.start() + attached_match.end()

    # Second try: for proper nouns, try last name only
    parts = attached.split()
    if len(parts) > 1:
        last_name = parts[-1]
        pattern = re.compile(r'\b' + re.escape(last_name) + r'\b', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.end()

    return None


def strip_references(text: str) -> str:
    """Remove all reference markers from text.

    Useful for comparison or when references need to be completely removed.

    Args:
        text: Text with references.

    Returns:
        Text without references.
    """
    result = REFERENCE_PATTERN.sub('', text)
    result = CITATION_PATTERN.sub('', result)
    result = re.sub(r'  +', ' ', result)
    return result
