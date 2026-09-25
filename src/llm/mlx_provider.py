"""MLX-based LLM provider for text generation.

Uses the same Qwen model as the LoRA pipeline for consistency.
This allows the entire pipeline to be self-contained without external services.
"""

import json
import re
import time
from typing import Optional, Callable
from pathlib import Path

from ..utils.logging import get_logger
from ..utils.prompts import load_prompt

logger = get_logger(__name__)


def _load_mlx_config() -> dict:
    """Load MLX config from config.json."""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("llm", {}).get("providers", {}).get("mlx", {})
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}

# Check MLX availability
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Install with: pip install mlx mlx-lm")


class MLXGenerator:
    """MLX-based text generator using Qwen model.

    Can be used for:
    - Neutralizing author text (converting to plain English)
    - Any other text generation tasks in the pipeline

    Configuration is loaded from config.json under llm.providers.mlx:
        {
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9
        }

    Example:
        generator = MLXGenerator()
        neutral = generator.generate(
            prompt="Convert to plain English: ...",
            max_tokens=200,
        )
    """

    DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the MLX generator.

        Args:
            model_name: Model to use (from config or default).
            temperature: Generation temperature (from config or 0.3).
            top_p: Top-p sampling parameter (from config or 0.9).
            max_tokens: Default max tokens (from config or 512).
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        # Load config defaults
        config = _load_mlx_config()

        self.model_name = model_name or config.get("model", self.DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.3)
        self.top_p = top_p if top_p is not None else config.get("top_p", 0.9)
        self.default_max_tokens = max_tokens if max_tokens is not None else config.get("max_tokens", 512)

        # Detect if this is a base model (no chat template)
        self.is_base_model = "instruct" not in self.model_name.lower() and "chat" not in self.model_name.lower()

        logger.info(f"MLX config: model={self.model_name}, base_model={self.is_base_model}, temp={self.temperature}")

        # Lazy load model
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is not None:
            return

        logger.debug(f"Loading MLX model: {self.model_name}")
        self._model, self._tokenizer = load(self.model_name)
        logger.debug("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            max_tokens: Maximum tokens to generate (from config if not specified).
            system_prompt: Optional system prompt.
            temperature: Override temperature for this call.

        Returns:
            Generated text.
        """
        self._ensure_loaded()
        max_tokens = max_tokens or self.default_max_tokens

        # For base models, use raw text completion
        if self.is_base_model:
            # Build a simple prompt format for base models
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                formatted_prompt = prompt
        else:
            # Build messages for instruct models
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Create sampler
        temp = temperature if temperature is not None else self.temperature
        sampler = make_sampler(temp=temp, top_p=self.top_p)

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        return response.strip()

    def unload(self):
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")


# Neutral output must stay within these bounds of the source word count.
MIN_RTT_LENGTH_RATIO = 0.6
MAX_RTT_LENGTH_RATIO = 1.6

_PLACEHOLDER_RE = re.compile(r'(?<![A-Za-z0-9])_*ENT(\d+)_*(?![A-Za-z0-9])', re.IGNORECASE)


def has_placeholder_residue(text: str) -> bool:
    """True if an entity placeholder (possibly mangled) survived restoration."""
    return bool(_PLACEHOLDER_RE.search(text))


def _is_sentence_initial(text: str, pos: int) -> bool:
    """True if the word at ``pos`` opens a sentence, line, quote or bracket."""
    before = text[:pos].rstrip(' \t')
    return not before or before[-1] in '.!?\n"\'(“‘:[' 


_BRACKET_MARKER_RE = re.compile(r'^\s*\[(\d+)\]\s*(.*)$')
_PLAIN_MARKER_RE = re.compile(r'^\s*(\d+)[.):]\s+(.*)$')


def parse_numbered_response(response: str, expected: int) -> dict:
    """Parse a batched "[n] text" response into {n: text}.

    Only in-range markers count. When the model used "[n]" anywhere, plain
    "n." lines are treated as text; otherwise a plain marker must be the next
    number in sequence. This keeps lines like "1918 was..." or "2. a list item"
    inside the item they belong to.
    """
    lines = response.strip().split('\n')
    use_brackets = any(
        (m := _BRACKET_MARKER_RE.match(line)) and 1 <= int(m.group(1)) <= expected
        for line in lines
    )

    items: dict = {}
    current = None
    for line in lines:
        match = (_BRACKET_MARKER_RE if use_brackets else _PLAIN_MARKER_RE).match(line)
        if match:
            num = int(match.group(1))
            in_range = 1 <= num <= expected and num not in items
            if in_range and (use_brackets or num == (current or 0) + 1):
                current = num
                items[num] = [match.group(2)] if match.group(2) else []
                continue
        if current is not None and line.strip():
            items[current].append(line.strip())

    parsed = {num: ' '.join(parts).strip() for num, parts in items.items()}
    return {num: text for num, text in parsed.items() if text}


class BaseRTTNeutralizer:
    """Shared logic for RTT neutralizers (entity masking + monotone flattening).

    Subclasses implement the translation step: MLX does local English→Mandarin→English,
    DeepSeek does a single API call with an equivalent prompt. Everything before and
    after translation — entity preservation and sentence flattening — lives here.
    """

    # Common English words that look like proper nouns after capitalization at a
    # sentence boundary. Skipping them prevents mangling ordinary text.
    _SKIP_WORDS = frozenset({
        'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'And', 'But', 'Or',
        'It', 'He', 'She', 'They', 'We', 'I', 'My', 'His', 'Her', 'Their',
        'This', 'That', 'These', 'Those', 'What', 'When', 'Where', 'Who',
        'How', 'Why', 'If', 'Then', 'Now', 'Here', 'There', 'So', 'Yet',
        'From', 'With', 'Into', 'Upon', 'Through', 'About', 'After', 'Before',
        'During', 'Within', 'Without', 'Between', 'Among', 'Against', 'Beyond',
        'Such', 'Each', 'Every', 'Some', 'Many', 'Most', 'Other', 'Another',
        'Both', 'All', 'Any', 'No', 'Not', 'Only', 'Just', 'Even', 'Still',
        'Already', 'Always', 'Never', 'Perhaps', 'Indeed', 'Thus', 'Hence',
        'Therefore', 'Moreover', 'Furthermore', 'Meanwhile', 'Otherwise',
        'Nonetheless', 'Nevertheless', 'However', 'Although', 'Though', 'While',
        'Because', 'Since', 'Unless', 'Until', 'Whether', 'Whenever', 'Wherever',
        'One', 'Two', 'Three', 'Four', 'Five', 'First', 'Second', 'Third',
    })

    def _extract_entities(self, text: str) -> tuple:
        """Mask proper nouns with __ENT{n}__ placeholders.

        Preserves names like "Jervas Dudley", "New England", "Squire Brewster"
        through the RTT process so they survive translation.

        Returns:
            (masked_text, entity_map) where entity_map is {placeholder: original}
        """
        import re

        entity_map: dict = {}
        counter = [0]

        def make_replacer():
            def replace_entity(match):
                word = match.group(1)
                if word.startswith('__ENT'):
                    return match.group(0)
                # Leading function words ("Then Kant") are not part of the name;
                # masking them would copy them verbatim into the neutral text.
                offset = 0
                for token in re.finditer(r'\S+', word):
                    if token.group() not in self._SKIP_WORDS:
                        offset = token.start()
                        break
                else:
                    return match.group(0)
                name = word[offset:]
                name_pos = match.start(1) + offset
                if ' ' not in name and _is_sentence_initial(match.string, name_pos):
                    # A lone capitalized word at a sentence start is ordinary
                    # capitalization, not a proper noun.
                    return match.group(0)
                placeholder = f"__ENT{counter[0]}__"
                entity_map[placeholder] = name
                counter[0] += 1
                prefix = match.group(0)[:name_pos - match.start(0)]
                return prefix + placeholder
            return replace_entity

        masked = text

        # Multi-word proper nouns (2-3 consecutive capitalized words)
        masked = re.sub(
            r'(?:^|(?<=[.!?,;:\s"\'(]))([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?=[\s.,;:!?\'")\-]|$)',
            make_replacer(),
            masked
        )

        # Single capitalized words. Sentence-initial ones are skipped in the
        # replacer, since the lookbehind alone can't see a preceding period.
        masked = re.sub(
            r'(?<=[,;:\s"\'(])([A-Z][a-z]{2,})(?=[\s.,;:!?\'")\-]|$)',
            make_replacer(),
            masked
        )

        return masked, entity_map

    def _restore_entities(self, text: str, entity_map: dict) -> str:
        """Restore original entities from __ENT{n}__ placeholders."""
        if not entity_map:
            return text

        # The translation step often mangles placeholders (__ENT0_, ENT0,
        # __ent0__), so match loosely on the number.
        def restore(match):
            return entity_map.get(f"__ENT{match.group(1)}__", match.group(0))

        return _PLACEHOLDER_RE.sub(restore, text)

    def _monotone_flatten(self, text: str) -> str:
        """Flatten text into uniform short sentences (rule-based, no LLM).

        Creates "The Monotone" — boring, repetitive input that maximizes delta
        from the stylized output. Breaks at sentence boundaries, semicolons,
        em-dashes, and conjunctions when sentences exceed 15 words. Targets
        8-15 words per output sentence.
        """
        import re

        # Asides become ordinary clauses. Deleting them (the old behaviour)
        # dropped content, so the styled target said things the input lacked.
        text = re.sub(r'\s*\(([^)]*)\)\s*', r', \1, ', text)
        text = re.sub(r'\s*[—–]\s*', ', ', text)
        text = re.sub(r'\s*,(\s*,)+\s*', ', ', text)
        text = re.sub(r'\s*,\s*([.!?;:])', r'\1', text)
        text = re.sub(r'^\s*,\s*', '', text)

        conjunctions = {'and', 'but', 'or', 'yet', 'so', 'however', 'although', 'while', 'whereas'}
        conj_pattern = r'\s*,?\s*\b(and|but|or|yet|so|however|although|while|whereas)\b\s*'

        def finish(segment: str) -> str:
            segment = segment.strip(' ,')
            if not segment.endswith(('.', '!', '?')):
                segment += '.'
            return segment[0].upper() + segment[1:]

        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            for part in re.split(r'\s*;\s*', sent):
                part = part.strip(' ,')
                if not part:
                    continue

                if len(part.split()) <= 15:
                    result.append(finish(part))
                    continue

                # Split long parts at conjunctions. The conjunction stays at the
                # head of the following clause, and clauses under 3 words are
                # folded back into the previous one rather than thrown away.
                segments = []
                conj = None
                for piece in re.split(conj_pattern, part, flags=re.IGNORECASE):
                    piece = piece.strip(' ,')
                    if not piece:
                        continue
                    if piece.lower() in conjunctions:
                        conj = piece.lower()
                        continue
                    clause = f"{conj} {piece}" if conj else piece
                    conj = None
                    if segments and len(piece.split()) < 3:
                        segments[-1] = f"{segments[-1].rstrip('.!?')}, {clause}"
                    else:
                        segments.append(clause)
                if conj and segments:
                    segments[-1] = f"{segments[-1]} {conj}"
                result.extend(finish(seg) for seg in segments)

        return ' '.join(result) if result else text


class RTTNeutralizer(BaseRTTNeutralizer):
    """Round-Trip Translation neutralizer using local MLX model.

    Uses Qwen2.5-3B-Instruct for English → Mandarin → English translation
    to strip style while preserving facts. The HSK constraint forces simple
    vocabulary, and the grammar distance flattens syntax.

    Configuration loaded from config.json under llm.providers.mlx_rtt.

    Example:
        neutralizer = RTTNeutralizer()
        neutral = neutralizer.neutralize(
            "The eldritch horror lurked in cyclopean shadows."
        )
        # -> "The strange monster hid in the big shadows."
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the RTT neutralizer.

        Args:
            model_name: Override model (defaults to config mlx_rtt.model).
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        # Load RTT-specific config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        rtt_config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                rtt_config = config.get("llm", {}).get("providers", {}).get("mlx_rtt", {})
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")

        self.model_name = model_name or rtt_config.get("model", "mlx-community/Qwen2.5-3B-Instruct-4bit")
        self.max_tokens = rtt_config.get("max_tokens", 512)
        self.temperature = rtt_config.get("temperature", 0.1)
        self.top_p = rtt_config.get("top_p", 0.9)

        logger.info(f"RTT neutralizer using: {self.model_name}")

        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Ensure model is loaded and warmed up."""
        if self._model is not None:
            return

        logger.info(f"Loading RTT model: {self.model_name}")
        self._model, self._tokenizer = load(self.model_name)

        # Create cached sampler (reused across calls)
        self._sampler = make_sampler(temp=self.temperature, top_p=self.top_p)

        # Warm up the model with a short generation (compiles MLX graphs)
        logger.info("Warming up model...")
        _ = generate(
            self._model,
            self._tokenizer,
            prompt="Hello",
            max_tokens=5,
            sampler=self._sampler,
        )
        logger.info("RTT model ready")

    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        """Generate using Qwen2.5 chat format."""
        self._ensure_loaded()

        # Build Qwen2.5 chat format
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self._sampler,  # Use cached sampler
        )

        return response.strip()

    def _rtt_once(self, text: str, word_count: int) -> Optional[str]:
        """Single pass through RTT: English → Mandarin → Plain English.

        Step 1 ("Acid Bath") strips Victorian/Lovecraftian syntax via HSK5
        Mandarin. Step 2 ("Flattener") emits clinical SVO English. Code fences
        are stripped; Chinese residue marks a failure. Length validation is
        left to the caller so that long single-sentence chunks (called via
        _do_neutralize) aren't rejected by the <100-word variance check.
        """
        max_mandarin_tokens = min(int(word_count * 2), 512)
        mandarin = self._generate(
            system=load_prompt("rtt_to_mandarin"),
            user=f"Translate to simple Mandarin:\n\n{text}",
            max_tokens=max_mandarin_tokens,
        )
        if not mandarin or len(mandarin) < 10:
            logger.debug("RTT Step 1 failed: empty Mandarin")
            return None

        max_english_tokens = min(int(word_count * 1.5) + 20, 400)
        english = self._generate(
            system=load_prompt("rtt_to_english"),
            user=f"Translate to simple English:\n\n{mandarin}",
            max_tokens=max_english_tokens,
        )
        if not english or len(english) < 10:
            logger.debug("RTT Step 2 failed: empty English")
            return None

        english = english.strip()
        english = re.sub(r'^```\w*\n?', '', english)
        english = re.sub(r'\n?```$', '', english)

        if re.search(r'[\u4e00-\u9fff]', english):
            logger.debug("RTT Step 2 failed: output contains Chinese")
            return None

        return english

    def neutralize(
        self,
        text: str,
        max_retries: int = 2,
        monotone: bool = False,
        _outer_masked: bool = False,
    ) -> Optional[str]:
        """Neutralize text via round-trip translation through Mandarin.

        Preserves proper nouns by masking them before RTT and restoring after.

        Args:
            text: Text to neutralize.
            max_retries: Number of retry attempts.
            monotone: If True, apply aggressive flattening to create
                uniform sentence lengths (for training burstiness).

        Returns:
            Neutralized text, or None if failed.
        """
        masked_text, entity_map = self._extract_entities(text)
        if entity_map:
            logger.debug(f"Masked {len(entity_map)} entities: {list(entity_map.values())[:5]}...")

        word_count = len(text.split())

        # For very long texts (300+ words), split into chunks
        if word_count > 300:
            result = self._neutralize_chunked(masked_text, max_retries, monotone)
            if result:
                result = self._restore_entities(result, entity_map)
                if not has_placeholder_residue(result):
                    return result
            return None

        for attempt in range(max_retries):
            try:
                english = self._rtt_once(masked_text, word_count)
                if english is None:
                    continue

                # Length validation (only applied here — long single-sentence
                # chunks go via _do_neutralize and skip this check).
                neutral_words = len(english.split())
                if neutral_words < 3:
                    logger.debug(f"RTT too short: {neutral_words} words (attempt {attempt + 1})")
                    continue

                max_diff = word_count * 1.0 if word_count < 100 else word_count * 2.0
                if abs(neutral_words - word_count) > max_diff:
                    logger.debug(
                        f"RTT length mismatch: {neutral_words} vs {word_count} (attempt {attempt + 1})"
                    )
                    continue

                if monotone:
                    english = self._monotone_flatten(english)

                if entity_map:
                    english = self._restore_entities(english, entity_map)
                # Chunks of a longer text still carry the caller's placeholders;
                # the caller checks residue after restoring them.
                if not _outer_masked and has_placeholder_residue(english):
                    logger.debug(f"RTT kept an unrestorable placeholder (attempt {attempt + 1})")
                    continue

                logger.debug(f"RTT success: {word_count} → {len(english.split())} words")
                return english

            except Exception as e:
                logger.warning(f"RTT attempt {attempt + 1} failed: {e}")

        return None

    def _do_neutralize(
        self,
        text: str,
        max_retries: int = 2,
        monotone: bool = False,
    ) -> Optional[str]:
        """Core RTT neutralization without chunking or length validation.

        Called directly for chunks that are >300 words but can't be split further
        (e.g., single sentences with no period boundaries). Entity masking is
        the caller's responsibility.
        """
        word_count = len(text.split())

        for attempt in range(max_retries):
            try:
                english = self._rtt_once(text, word_count)
                if english is None:
                    continue

                neutral_words = len(english.split())
                if neutral_words < 3:
                    continue

                if monotone:
                    english = self._monotone_flatten(english)

                logger.debug(f"RTT success: {word_count} → {len(english.split())} words")
                return english

            except Exception as e:
                logger.warning(f"RTT attempt {attempt + 1} failed: {e}")

        return None

    def _neutralize_chunked(
        self,
        text: str,
        max_retries: int,
        monotone: bool,
    ) -> Optional[str]:
        """Neutralize long text by splitting into ~150 word chunks.

        Args:
            text: Long text to neutralize (300+ words)
            max_retries: Retries per chunk
            monotone: Apply monotone flattening

        Returns:
            Combined neutralized text, or None if failed
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Group sentences into ~150 word chunks
        chunks = []
        current_chunk = []
        current_words = 0

        for sent in sentences:
            sent_words = len(sent.split())
            if current_words + sent_words > 150 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_words = sent_words
            else:
                current_chunk.append(sent)
                current_words += sent_words

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Neutralize each chunk
        results = []
        for chunk in chunks:
            chunk_words = len(chunk.split())
            if chunk_words > 300:
                # Chunk is still too long (e.g., single sentence with no periods)
                # Call _do_neutralize directly to avoid infinite recursion
                result = self._do_neutralize(chunk, max_retries=max_retries, monotone=False)
            else:
                # Short enough chunk — safe to call neutralize (won't recurse)
                result = self.neutralize(chunk, max_retries=max_retries, monotone=False, _outer_masked=True)
            if result:
                results.append(result)
            else:
                logger.debug(f"Chunk failed: {chunk_words} words")
                # Continue with other chunks

        if not results:
            return None

        combined = ' '.join(results)

        # Apply monotone flattening once at the end
        if monotone:
            combined = self._monotone_flatten(combined)

        return combined

class DeepSeekRTTNeutralizer(BaseRTTNeutralizer):
    """Round-Trip Translation neutralizer using DeepSeek API.

    Faster than local MLX for bulk processing. Supports batching multiple
    chunks in a single API call for efficiency.

    Configuration loaded from config.json under llm.providers.deepseek_rtt.
    """

    def __init__(self, batch_size: int = 5):
        """Initialize the DeepSeek RTT neutralizer.

        Args:
            batch_size: Number of texts to process in a single API call.
        """
        import os
        from ..config import LLMProviderConfig
        from .deepseek import DeepSeekProvider

        # Load config
        config_path = Path(__file__).parent.parent.parent / "config.json"
        rtt_config: dict = {}
        deepseek_config: dict = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                rtt_config = config.get("llm", {}).get("providers", {}).get("deepseek_rtt", {})
                deepseek_config = config.get("llm", {}).get("providers", {}).get("deepseek", {})
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")

        # Resolve API key: config ${VAR} or literal, then env var fallback.
        api_key = ""
        config_api_key = deepseek_config.get("api_key", "")
        if config_api_key.startswith("${") and config_api_key.endswith("}"):
            api_key = os.environ.get(config_api_key[2:-1], "")
        elif config_api_key:
            api_key = config_api_key
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Set it in config.json or as environment variable."
            )

        self.model = rtt_config.get("model", "deepseek-chat")
        self.max_tokens = rtt_config.get("max_tokens", 8192)
        self.temperature = rtt_config.get("temperature", 0.1)
        self.batch_size = batch_size or rtt_config.get("batch_size", 5)
        self.concurrent_batches = rtt_config.get("concurrent_batches", 4)

        # Route HTTP + retry + error handling through the standard DeepSeek provider
        # so this class owns RTT logic only (entity masking, Chinese detection, monotone
        # flattening, batch parsing) and stays out of the HTTP business.
        self._provider = DeepSeekProvider(
            LLMProviderConfig(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=rtt_config.get("timeout", 180),
            )
        )

        logger.debug(f"DeepSeek RTT: model={self.model}, batch_size={self.batch_size}, concurrent={self.concurrent_batches}")

    def _call_api_full(self, system: str, user: str, max_tokens: Optional[int] = None) -> tuple:
        """One chat completion via the DeepSeek provider -> (content, finish_reason)."""
        from ..models.base import Message, MessageRole
        response = self._provider._call_with_retry(
            [
                Message(role=MessageRole.SYSTEM, content=system),
                Message(role=MessageRole.USER, content=user),
            ],
            self.temperature,
            max_tokens or self.max_tokens,
            False,
        )
        return response.content.strip(), response.finish_reason

    def neutralize(
        self,
        text: str,
        max_retries: int = 2,
        monotone: bool = False,
    ) -> Optional[str]:
        """Neutralize a single text via DeepSeek RTT.

        Goes through the same batch prompt and checks as training data
        generation, so inference inputs look like training inputs.

        Returns:
            Neutralized text, or None if failed.
        """
        for attempt in range(max_retries):
            results = self._process_single_batch((0, [text]), monotone=monotone)
            if results:
                return results[0][1]
            logger.debug(f"DeepSeek RTT attempt {attempt + 1} failed")
        return None

    def _accept_item(self, source: str, text: str, entity_map: dict, monotone: bool) -> Optional[str]:
        """Clean one parsed item, or return None if it can't be trusted."""
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text).strip()
        if not text or re.search(r'[\u4e00-\u9fff]', text):
            return None

        # The prompt asks for similar length. Much shorter means content was
        # dropped; much longer means content was added.
        ratio = len(text.split()) / max(len(source.split()), 1)
        if ratio < MIN_RTT_LENGTH_RATIO or ratio > MAX_RTT_LENGTH_RATIO:
            logger.debug(f"RTT length ratio {ratio:.2f} out of range")
            return None

        if monotone:
            text = self._monotone_flatten(text)
        text = self._restore_entities(text, entity_map)
        if has_placeholder_residue(text):
            logger.debug("RTT output kept an unrestorable entity placeholder")
            return None
        return text

    def _process_single_batch(
        self,
        batch_info: tuple,
        monotone: bool = False,
    ) -> list:
        """Process a single batch of texts (for parallel execution).

        Args:
            batch_info: Tuple of (batch_start, batch_texts)
            monotone: If True, apply monotone flattening.

        Returns:
            List of (global_index, result) tuples. Items that failed a check
            are left out; the caller retries them.
        """
        batch_start, batch = batch_info

        masked_batch = []
        entity_maps = []
        for text in batch:
            masked, entity_map = self._extract_entities(text)
            masked_batch.append(masked)
            entity_maps.append(entity_map)

        batch_prompt = "Perform 'Chemical Dissolution' on each numbered text below.\n"
        batch_prompt += "Output each final neutral English text on a new line starting with the same [n] marker.\n"
        batch_prompt += "Keep all __ENT*__ placeholders unchanged.\n\n"
        for i, masked in enumerate(masked_batch):
            batch_prompt += f"[{i+1}] {masked}\n\n"

        total_words = sum(len(t.split()) for t in batch)
        max_tokens = min(int(total_words * 2) + 100, 8000)

        try:
            response, finish_reason = self._call_api_full(
                system=load_prompt("rtt_deepseek_batch"),
                user=batch_prompt,
                max_tokens=max_tokens,
            )
        except Exception as e:
            # No per-item fallback here: neutralize() and neutralize_batch()
            # already retry, and falling back to neutralize() would recurse.
            logger.warning(f"Batch {batch_start}-{batch_start + len(batch)} failed: {e}")
            return []

        items = parse_numbered_response(response, expected=len(batch))
        if finish_reason == "length" and items:
            # The last item was cut off mid-text.
            logger.debug("RTT batch response truncated, dropping last item")
            items.pop(max(items))

        results = []
        for num, text in items.items():
            idx = num - 1
            accepted = self._accept_item(batch[idx], text, entity_maps[idx], monotone)
            if accepted:
                results.append((batch_start + idx, accepted))
        return results

    def neutralize_batch(
        self,
        texts: list,
        monotone: bool = False,
        on_progress: Optional[Callable[[int, int], None]] = None,
        max_retries: int = 2,
    ) -> list:
        """Neutralize multiple texts using a queue-based parallel pipeline.

        Uses a fixed-size work queue that workers continuously pull from.
        Failed items go back into the queue for retry (up to max_retries).
        This ensures consistent throughput without waiting for stragglers.

        Args:
            texts: List of texts to neutralize.
            monotone: If True, apply monotone flattening.
            on_progress: Optional callback (processed, total).
            max_retries: Max retry attempts per item (default 2).

        Returns:
            List of neutralized texts (None for failures after all retries).
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading
        from queue import Queue, Empty

        results = [None] * len(texts)
        results_lock = threading.Lock()

        # Track completion and retries
        completed = [0]  # Successfully processed
        final_failures = [0]  # Failed after all retries
        retry_counts = {}  # idx -> retry count
        completed_lock = threading.Lock()

        # Work queue: (original_idx, text, attempt)
        work_queue = Queue()
        for idx, text in enumerate(texts):
            work_queue.put((idx, text, 0))
            retry_counts[idx] = 0

        total_items = len(texts)

        logger.info(f"Processing {len(texts)} texts with queue-based pipeline "
                   f"(batch_size={self.batch_size}, workers={self.concurrent_batches}, max_retries={max_retries})")

        def worker():
            """Worker that pulls batches from queue and processes them."""
            stagger_delay = 0.1

            while True:
                # Collect a batch from the queue
                batch_items = []
                try:
                    # Try to fill a batch
                    while len(batch_items) < self.batch_size:
                        try:
                            item = work_queue.get(timeout=0.5)
                            batch_items.append(item)
                        except Empty:
                            break

                    if not batch_items:
                        # Check if we should exit
                        with completed_lock:
                            done = completed[0] + final_failures[0]
                            if done >= total_items:
                                break
                            # Queue might be temporarily empty, keep waiting
                            if work_queue.empty():
                                # Double-check completion
                                time.sleep(0.1)
                                if work_queue.empty() and completed[0] + final_failures[0] >= total_items:
                                    break
                        continue

                    # Stagger to avoid rate limit spikes
                    time.sleep(stagger_delay)

                    # Process the batch
                    batch_texts = [text for _, text, _ in batch_items]
                    batch_indices = [idx for idx, _, _ in batch_items]

                    # Build batch for _process_single_batch (it expects (start_idx, texts))
                    # We'll process and map results back manually
                    batch_info = (0, batch_texts)
                    batch_results = self._process_single_batch(batch_info, monotone=monotone)

                    # Map results back: batch_results is list of (local_idx, result)
                    result_map = {local_idx: result for local_idx, result in batch_results}

                    # Process each item in batch
                    for i, (orig_idx, text, attempt) in enumerate(batch_items):
                        result = result_map.get(i)

                        if result:
                            # Success
                            with results_lock:
                                results[orig_idx] = result
                            with completed_lock:
                                completed[0] += 1
                                if on_progress:
                                    on_progress(completed[0], total_items)
                        else:
                            # Failure - retry or mark as final failure
                            with completed_lock:
                                retry_counts[orig_idx] += 1
                                if retry_counts[orig_idx] < max_retries:
                                    # Put back in queue for retry
                                    work_queue.put((orig_idx, text, retry_counts[orig_idx]))
                                else:
                                    # Max retries exhausted
                                    final_failures[0] += 1
                                    logger.debug(f"[{orig_idx}] Failed after {max_retries} attempts")

                except Exception as e:
                    logger.warning(f"Worker error: {e}")
                    # Put items back in queue with incremented retry count
                    for item in batch_items:
                        orig_idx, text, retry_count = item
                        retry_counts[orig_idx] += 1
                        if retry_counts[orig_idx] >= max_retries:
                            with completed_lock:
                                final_failures[0] += 1
                                logger.debug(f"[{orig_idx}] Failed after {max_retries} attempts (worker error)")
                        else:
                            work_queue.put((orig_idx, text, retry_counts[orig_idx]))

        # Start workers
        with ThreadPoolExecutor(max_workers=self.concurrent_batches) as executor:
            futures = [executor.submit(worker) for _ in range(self.concurrent_batches)]

            # Wait for all workers to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Worker thread error: {e}")

        logger.info(f"Queue complete: {completed[0]} succeeded, {final_failures[0]} failed after retries")
        return results


def create_rtt_neutralizer(provider: str = None, batch_size: int = None):
    """Factory function to create the appropriate RTT neutralizer.

    Args:
        provider: 'mlx' or 'deepseek'. If None, reads from config.json.
        batch_size: Batch size for DeepSeek (ignored for MLX).

    Returns:
        RTTNeutralizer or DeepSeekRTTNeutralizer instance.
    """
    if provider is None:
        # Load from config using type-safe config system
        try:
            from ..config import load_config
            config = load_config()
            provider = config.llm.provider.rtt
        except Exception:
            provider = "deepseek"

    if provider == "mlx":
        return RTTNeutralizer()
    elif provider == "deepseek":
        return DeepSeekRTTNeutralizer(batch_size=batch_size)
    else:
        raise ValueError(f"Unknown RTT provider: {provider}")
