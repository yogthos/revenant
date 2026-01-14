"""Base class for style generators with shared utilities.

This module provides the abstract base class and common functionality
shared between MLX and PyTorch backends.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from ..utils.logging import get_logger

logger = get_logger(__name__)


def generate_style_tag(text: str) -> str:
    """Generate a structural style tag based on text analysis.

    Tags describe the structural features the model should produce:
    - Length pattern: Short & Punchy, Varied Lengths, Long & Flowing
    - Complexity: Simple Syntax, Complex Syntax, Baroque Syntax

    Example: [STYLE: Varied Lengths | Complex Syntax]
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return "[STYLE: Medium Length | Simple Syntax]"

    # Calculate sentence lengths
    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)

    # Calculate variance
    if len(lengths) > 1:
        variance = sum((x - avg_length) ** 2 for x in lengths) / len(lengths)
        std_dev = variance ** 0.5
    else:
        std_dev = 0

    # Detect complex syntax markers
    complex_markers = [';', '—', '--', ':', '(', ')']
    has_complex = any(m in text for m in complex_markers)

    # Detect literary connectives
    connective_words = ['however', 'although', 'yet', 'moreover', 'furthermore',
                        'nevertheless', 'whilst', 'whereas']
    has_literary_connectives = any(w in text.lower() for w in connective_words)

    # Determine length pattern
    if avg_length < 12:
        length_tag = "Short & Punchy"
    elif avg_length > 25:
        length_tag = "Long & Flowing"
    elif std_dev > 8:
        length_tag = "Varied Lengths"
    else:
        length_tag = "Medium Length"

    # Determine complexity
    if has_complex and has_literary_connectives:
        complexity_tag = "Baroque Syntax"
    elif has_complex:
        complexity_tag = "Complex Syntax"
    else:
        complexity_tag = "Simple Syntax"

    return f"[STYLE: {length_tag} | {complexity_tag}]"


@dataclass
class GenerationConfig:
    """Configuration for style generation.

    Values are loaded from config.json lora_adapters section. Use from_config() to create.
    """

    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.92
    min_p: float = 0.05
    repetition_penalty: float = 1.15
    scale: float = 1.0  # LoRA adapter scale
    skip_cleaning: bool = False  # If True, return raw output without cleaning

    @classmethod
    def from_config(cls, adapter_path: Optional[str] = None) -> "GenerationConfig":
        """Create GenerationConfig from config.json settings for a specific adapter.

        Args:
            adapter_path: Path to the LoRA adapter. If provided, loads adapter-specific
                         settings from generation.lora_adapters[path]. If None, uses defaults.

        Returns:
            GenerationConfig with settings from config or defaults.
        """
        try:
            from ..config import get_adapter_config
            adapter_config = get_adapter_config(adapter_path)
            return cls(
                max_tokens=adapter_config.max_tokens,
                temperature=adapter_config.temperature,
                top_p=adapter_config.top_p,
                min_p=adapter_config.min_p,
                repetition_penalty=adapter_config.repetition_penalty,
                scale=adapter_config.scale,
            )
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            return cls()


class BaseStyleGenerator(ABC):
    """Abstract base class for style generators.

    Provides common functionality for cleaning model output and ensuring
    complete sentences. Subclasses must implement generate() and unload().
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize base generator.

        Args:
            config: Generation configuration.
        """
        self.config = config or GenerationConfig()

    @abstractmethod
    def generate(
        self,
        content: str,
        author: str,
        max_tokens: Optional[int] = None,
        target_words: Optional[int] = None,
        structural_guidance: Optional[str] = None,
        raw_prompt: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate styled text from content.

        Args:
            content: What to express (neutral text to restyle).
            author: Author name (used in prompt).
            max_tokens: Override for max tokens (defaults to config).
            target_words: Target word count for output.
            structural_guidance: Formatted structural guidance (rhythm, punctuation hints).
            raw_prompt: If True, use content directly as prompt without formatting.
            temperature: Override for sampling temperature (defaults to config).

        Returns:
            Generated text in the author's style.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model to free memory."""
        pass

    def _clean_response(self, response: str) -> str:
        """Clean model output of obvious garbage only.

        Removes:
        - ### markers and everything after (model repetition boundary)
        - Non-ASCII garbage (Thai, Cyrillic, Chinese characters)
        - <think> tags
        - Training format markers

        Does NOT aggressively detect repetition - preserves content.

        Args:
            response: Raw model output.

        Returns:
            Cleaned response text.
        """
        original_len = len(response.split())

        # 1. Stop at ### markers (model uses these before repeating)
        if '###' in response:
            before = response.split('###')[0].strip()
            removed = original_len - len(before.split())
            if removed > 10:
                logger.debug(f"### marker removed {removed} words")
            response = before

        # 2. Stop at training format markers and obvious garbage patterns
        training_markers = [
            "[NEUTRAL INPUT]:",
            "[NEUTRAL INPUT]",
            "_OUTPUT]:",  # Any [AUTHOR_OUTPUT]: marker
            "\n\nRewrite the following",
            "\n\n---",
            "_NOTE:",
            ".DebugLine",  # Model garbage
            ".debugLine",  # Model garbage (lowercase)
            "DebugLine:",  # Model garbage
        ]
        for marker in training_markers:
            if marker in response:
                before = response.split(marker)[0].strip()
                removed = len(response.split()) - len(before.split())
                if removed > 10:
                    logger.debug(f"Training marker '{marker}' removed {removed} words")
                response = before

        # 2b. Remove obvious numeric/symbol garbage sequences
        garbage_seq_pattern = r'[\d!@#$%^&*()_+{}|:"<>?,./;\'\[\]\\=\-]{10,}'
        response = re.sub(garbage_seq_pattern, '', response)

        # 3. Remove <think>...</think> blocks (Qwen3 thinking mode)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        if '<think>' in response:
            response = response.split('<think>')[0]

        # 4. Remove non-ASCII garbage characters (Thai, Cyrillic, Arabic, Chinese, etc.)
        garbage_ranges = r'[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0E00-\u0E7F\uAC00-\uD7AF\u3040-\u30FF\u4E00-\u9FFF\u0080-\u009F\u3000-\u303F\uFF00-\uFFEF]'

        # Stop at first garbage character sequence
        match = re.search(garbage_ranges + r'+', response)
        if match:
            before = response[:match.start()].strip()
            removed = len(response.split()) - len(before.split())
            if removed > 10:
                logger.debug(f"Non-ASCII garbage at pos {match.start()} removed {removed} words")
            response = before

        # 5. Stop at Chinese punctuation that often precedes garbage
        for stop_char in ['：']:
            if stop_char in response:
                before = response.split(stop_char)[0].strip()
                removed = len(response.split()) - len(before.split())
                if removed > 10:
                    logger.debug(f"Chinese punctuation '{stop_char}' removed {removed} words")
                response = before

        # 6. Clean up artifacts
        response = re.sub(r'\s{2,}', ' ', response)  # Multiple spaces
        response = re.sub(r'\n{3,}', '\n\n', response)  # Multiple newlines

        response = response.strip()

        # 6b. Fix broken atmospheric openings
        response = self._fix_broken_atmospheric_phrases(response)

        # 7. Remove sentences containing fiction-specific markers (hallucinations)
        fiction_markers = [
            r'\barkham\b', r'\bcthulhu\b', r'\bnecronomicon\b', r'\bmiskatonic\b',
            r'\binnsmouth\b', r'\bdunwich\b', r'\bshoggoth\b', r'\byog-sothoth\b',
            r'\bazathoth\b', r'\bnyarlathotep\b', r'\br\'lyeh\b', r'\bdagon\b',
            r'\bmy letters to you\b', r'\bin my letters\b', r'\bi have seen\b',
            r'\bi have written\b', r'\bi cannot recall\b', r'\bwhose names i\b',
        ]
        fiction_pattern = re.compile('|'.join(fiction_markers), re.IGNORECASE)

        # Split into sentences and filter
        sentences = re.split(r'(?<=[.!?])\s+', response)
        clean_sentences = []
        for sent in sentences:
            if not fiction_pattern.search(sent):
                clean_sentences.append(sent)
            else:
                logger.debug(f"Removed fiction hallucination: {sent[:80]}...")

        if clean_sentences:
            response = ' '.join(clean_sentences)

        response = response.strip()

        # 8. Ensure text ends with a complete sentence
        response = self._ensure_complete_sentences(response)

        return response

    def _ensure_complete_sentences(self, text: str) -> str:
        """Ensure text ends with a complete sentence.

        If text ends mid-sentence, truncate to the last complete sentence.
        """
        text = text.strip()
        if not text:
            return text

        # If already ends with sentence punctuation, we're good
        if text[-1] in '.!?':
            return text

        # Find the last complete sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter to only complete sentences (end with punctuation)
        complete = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent[-1] in '.!?':
                complete.append(sent)

        if complete:
            return ' '.join(complete)

        # No complete sentences - try to add period if it looks like a complete thought
        if len(text.split()) > 10 and text[-1] not in ',:;':
            return text + '.'

        return text

    def _fix_broken_atmospheric_phrases(self, text: str) -> str:
        """Fix broken prepositional phrase openings using spaCy parsing.

        Detects patterns where a prepositional phrase ending in "of" is followed
        by what should be a subject+verb (not the object of "of").

        Example: "Beneath the weight of technology advances"
        - "technology" looks like object of "of" but "advances" is a verb
        - This indicates "technology advances" is actually a subject+verb pair
        - Fix: Remove the prepositional phrase -> "Technology advances"

        Args:
            text: Text to fix.

        Returns:
            Text with broken prepositional phrases removed.
        """
        try:
            from ..utils.nlp import get_nlp
            nlp = get_nlp()
        except Exception:
            # If spaCy not available, return unchanged
            return text

        # Process the text
        doc = nlp(text)

        # Check if text starts with a prepositional phrase ending in "of"
        first_sent = list(doc.sents)[0] if doc.sents else None
        if not first_sent:
            return text

        tokens = list(first_sent)
        if len(tokens) < 4:
            return text

        # Find "of" in the first ~10 tokens
        of_idx = None
        for i, tok in enumerate(tokens[:10]):
            if tok.text.lower() == 'of':
                of_idx = i
                break

        if of_idx is None or of_idx < 2:
            return text

        # Check if what follows "of" is a noun followed by a verb
        if of_idx + 2 < len(tokens):
            next_tok = tokens[of_idx + 1]
            after_next = tokens[of_idx + 2]

            # Pattern: "of" + NOUN + VERB (the noun should be subject, not object of "of")
            if (next_tok.pos_ in ('NOUN', 'PROPN') and
                after_next.pos_ == 'VERB' and
                after_next.dep_ in ('ROOT', 'ccomp', 'advcl')):

                # The phrase before "of" + noun is broken - remove it
                start_char = next_tok.idx
                if start_char >= len(text) - 1:
                    return text  # Bounds check
                fixed_text = text[start_char].upper() + text[start_char + 1:]
                logger.debug(f"Fixed broken prepositional opening at '{next_tok.text}'")
                return fixed_text

        return text
