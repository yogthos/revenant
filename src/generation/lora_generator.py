"""LoRA-based style transfer generator using MLX.

This module provides fast style transfer using a LoRA-adapted model.
Style is baked into the adapter weights, eliminating the need for:
- Multi-candidate evolutionary search
- Example-based prompting
- Statistical style verification

Performance target: ~5-10 seconds per paragraph generation.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

from ..utils.logging import get_logger
from ..utils.prompts import format_prompt
from .base_generator import (
    BaseStyleGenerator,
    GenerationConfig,
    generate_style_tag,
    split_legacy_prompt,
)

logger = get_logger(__name__)


# Check MLX availability at module level
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. LoRA generation will not work.")


@dataclass
class AdapterMetadata:
    """Metadata about a LoRA adapter."""

    author: str
    base_model: str
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    training_examples: int = 0
    # LlamaFactory template and enable_thinking used in training (see
    # base_generator.render_chat_prompt). Written by convert_peft_to_mlx.py.
    template: str = ""
    enable_thinking: Optional[bool] = None
    persona_turn: str = ""

    @classmethod
    def from_file(cls, path: Path) -> "AdapterMetadata":
        """Load metadata from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            author=data.get("author", "Unknown"),
            base_model=data.get("base_model", "mlx-community/Qwen3-8B-Base-bf16"),
            lora_rank=data.get("lora_rank", 16),
            lora_alpha=data.get("lora_alpha", 32),
            epochs=data.get("epochs", 3),
            training_examples=data.get("training_examples", 0),
            template=data.get("template", ""),
            enable_thinking=data.get("enable_thinking"),
            persona_turn=data.get("persona_turn", ""),
        )


def stack_lora_adapters(adapters, xp=None):
    """Combine LoRA adapters into one of higher rank.

    ``adapters`` is a list of ``(weights, trained_scale, scale)``, where
    weights maps ``<module>.lora_a`` (in, r) and ``<module>.lora_b`` (r, out).
    Stacking A side by side and the scaled B blocks on top of each other gives
    ``x @ A @ B == sum(trained_scale * scale * x @ A_i @ B_i)`` exactly, so the
    result is loaded with a LoRA scale of 1.0. Summing the A and B matrices
    separately would add cross terms between adapters.

    Returns (weights, total_rank).
    """
    if xp is None:
        import numpy as xp
    modules = sorted({k.rsplit(".lora_", 1)[0] for w, _, _ in adapters for k in w})
    ranks = []
    for weights, _, _ in adapters:
        a = next(v for k, v in weights.items() if k.endswith(".lora_a"))
        if a.ndim != 2:
            raise ValueError("Only 2-D LoRA weights can be blended")
        ranks.append(a.shape[1])

    stacked = {}
    for module in modules:
        dims = next((w[f"{module}.lora_a"].shape[0], w[f"{module}.lora_b"].shape[1], w[f"{module}.lora_a"].dtype)
                    for w, _, _ in adapters if f"{module}.lora_a" in w)
        in_dim, out_dim, dtype = dims
        a_blocks, b_blocks = [], []
        for (weights, trained_scale, scale), rank in zip(adapters, ranks):
            if f"{module}.lora_a" in weights:
                a_blocks.append(weights[f"{module}.lora_a"])
                b_blocks.append(weights[f"{module}.lora_b"] * (trained_scale * scale))
            else:
                a_blocks.append(xp.zeros((in_dim, rank), dtype=dtype))
                b_blocks.append(xp.zeros((rank, out_dim), dtype=dtype))
        stacked[f"{module}.lora_a"] = xp.concatenate(a_blocks, axis=1)
        stacked[f"{module}.lora_b"] = xp.concatenate(b_blocks, axis=0)
    return stacked, sum(ranks)


@dataclass
class AdapterSpec:
    """Specification for a single LoRA adapter with its scale."""

    path: str
    scale: float = 1.0
    checkpoint: Optional[str] = None

    @classmethod
    def parse(cls, spec: str) -> "AdapterSpec":
        """Parse adapter spec from string format 'path:scale' or 'path'.

        Examples:
            'lora_adapters/sagan' -> AdapterSpec(path='lora_adapters/sagan', scale=1.0)
            'lora_adapters/sagan:0.5' -> AdapterSpec(path='lora_adapters/sagan', scale=0.5)
        """
        if ":" in spec:
            parts = spec.rsplit(":", 1)
            try:
                scale = float(parts[1])
                return cls(path=parts[0], scale=scale)
            except ValueError:
                # Colon was part of path (e.g., Windows path)
                return cls(path=spec, scale=1.0)
        return cls(path=spec, scale=1.0)


class LoRAStyleGenerator(BaseStyleGenerator):
    """Fast style transfer using LoRA-adapted model with MLX backend.

    Key advantages over prompted approach:
    - Style baked into weights (no examples needed in prompt)
    - Single forward pass (no evolutionary search)
    - Consistent voice (no mode collapse across calls)

    Example usage:
        generator = LoRAStyleGenerator(
            adapter_path="lora_adapters/sagan",
            config=GenerationConfig(temperature=0.7),
        )

        # Generate styled text
        output = generator.generate(
            content="The universe is vast. Stars are distant suns.",
            author="Carl Sagan",
        )
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model: str = "mlx-community/Qwen3-8B-Base-bf16",
        config: Optional[GenerationConfig] = None,
        checkpoint: Optional[str] = None,
        adapters: Optional[List[AdapterSpec]] = None,
    ):
        """Initialize the LoRA generator.

        Args:
            adapter_path: Path to LoRA adapter directory (for single adapter, backward compatible).
            base_model: Base model (overridden by adapter metadata if available).
            config: Generation configuration.
            checkpoint: Specific checkpoint file to use (e.g., "0000600_adapters.safetensors").
                       If provided, creates a temp directory with symlinks to use this checkpoint.
            adapters: List of AdapterSpec for multiple adapters. If provided, adapter_path is ignored.
                     Each adapter can have its own scale for blending multiple styles.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        super().__init__(config or GenerationConfig.from_config())
        self.base_model_name = base_model
        self.metadata: Optional[AdapterMetadata] = None
        self._temp_dirs: List[str] = []  # For checkpoint symlink directories

        # Handle adapter specification
        if adapters:
            # Multiple adapters mode
            self.adapters = adapters
            self.adapter_path = None
            self.checkpoint = None
        elif adapter_path:
            # Single adapter mode (backward compatible) - scale defaults to 1.0
            self.adapters = [
                AdapterSpec(path=adapter_path, scale=1.0, checkpoint=checkpoint)
            ]
            self.adapter_path = adapter_path
            self.checkpoint = checkpoint
        else:
            # Fused model mode (no adapter) — base_model is the fused model path
            self.adapters = []
            self.adapter_path = None
            self.checkpoint = None

        # Lazy load model
        self._model = None
        self._tokenizer = None

        # Built lazily on first generate() call: sentinel None = not-yet-built,
        # callable = the processor, False = empty bias map (nothing to apply).
        self._logit_bias_processor = None

        # Load metadata from first adapter if available
        if self.adapters:
            first_adapter_path = Path(self.adapters[0].path)
            metadata_path = first_adapter_path / "metadata.json"
            if metadata_path.exists():
                self.metadata = AdapterMetadata.from_file(metadata_path)
                self.base_model_name = self.metadata.base_model
                logger.info(f"Loaded adapter metadata: {self.metadata.author}")

    def _is_model_cached(self, model_name: str) -> bool:
        """Check if model is already downloaded in HuggingFace cache."""
        try:
            from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

            # Check for config.json as indicator the model is cached
            result = try_to_load_from_cache(model_name, "config.json")
            return result is not None and result is not _CACHED_NO_EXIST
        except Exception:
            return False

    def _setup_checkpoint_adapter(self, adapter_spec: AdapterSpec) -> str:
        """Create temp directory with symlinks for checkpoint loading.

        Args:
            adapter_spec: Adapter specification with path and checkpoint.

        Returns:
            Path to use as adapter_path (temp dir with symlinks).
        """
        adapter_dir = Path(adapter_spec.path)
        checkpoint_file = adapter_dir / adapter_spec.checkpoint

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="lora_checkpoint_")
        self._temp_dirs.append(temp_dir)
        temp_path = Path(temp_dir)

        # Symlink the checkpoint as adapters.safetensors
        (temp_path / "adapters.safetensors").symlink_to(checkpoint_file.resolve())

        # Symlink adapter_config.json if it exists
        config_file = adapter_dir / "adapter_config.json"
        if config_file.exists():
            (temp_path / "adapter_config.json").symlink_to(config_file.resolve())

        logger.info(f"Using checkpoint: {adapter_spec.checkpoint}")
        return temp_dir

    def _get_effective_adapter_path(self, adapter_spec: AdapterSpec) -> str:
        """Get effective adapter path, handling checkpoints.

        Args:
            adapter_spec: Adapter specification.

        Returns:
            Path to adapter directory (possibly temp dir with symlinks for checkpoints).
        """
        if adapter_spec.checkpoint:
            return self._setup_checkpoint_adapter(adapter_spec)
        return adapter_spec.path

    def _load_adapter_weights(self, adapter_path: str) -> dict:
        """Load adapter weights from safetensors file.

        Args:
            adapter_path: Path to adapter directory.

        Returns:
            Dictionary of weight name -> weight array.
        """
        import mlx.core as mx

        weights_file = Path(adapter_path) / "adapters.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"Adapter weights not found: {weights_file}")

        return mx.load(str(weights_file))

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is not None:
            return

        is_cached = self._is_model_cached(self.base_model_name)
        if is_cached:
            logger.debug(f"Loading model: {self.base_model_name}")
            # Suppress progress bars for cached models
            old_hf_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        else:
            logger.debug(f"Downloading model: {self.base_model_name}")
            old_hf_disable = None

        try:
            if self.adapters:
                self._load_with_adapters()
            else:
                self._model, self._tokenizer = load(self.base_model_name)
        finally:
            # Restore progress bar setting
            if is_cached:
                if old_hf_disable is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_hf_disable

        logger.debug("Model loaded successfully")

    def _load_with_adapters(self):
        """Load model with one or more LoRA adapters."""
        import mlx.core as mx

        if len(self.adapters) == 1:
            # Single adapter - use standard loading path
            adapter = self.adapters[0]
            effective_path = self._get_effective_adapter_path(adapter)

            logger.debug(f"With LoRA adapter: {adapter.path} (scale={adapter.scale})")
            self._model, self._tokenizer = load(
                self.base_model_name,
                adapter_path=effective_path,
            )

            # Apply scale if not 1.0
            if adapter.scale != 1.0:
                self._apply_lora_scale(adapter.scale)
        else:
            # Multiple adapters: stack them into one higher-rank LoRA.
            import re
            from mlx.utils import tree_flatten
            from mlx_lm.tuner.utils import linear_to_lora_layers

            logger.info(f"Loading {len(self.adapters)} adapters:")
            self._model, self._tokenizer = load(self.base_model_name)

            blend = []
            for adapter in self.adapters:
                logger.info(f"  - {adapter.path} (scale={adapter.scale})")
                path = self._get_effective_adapter_path(adapter)
                with open(Path(path) / "adapter_config.json") as f:
                    trained_scale = json.load(f)["lora_parameters"]["scale"]
                blend.append((self._load_adapter_weights(path), trained_scale, adapter.scale))

            weights, rank = stack_lora_adapters(blend, xp=mx)
            keys = sorted({re.sub(r"^.*?layers\.\d+\.", "", k.rsplit(".lora_", 1)[0]) for k in weights})
            linear_to_lora_layers(self._model, -1, {"rank": rank, "scale": 1.0, "dropout": 0.0, "keys": keys})
            missing = set(weights) - {k for k, _ in tree_flatten(self._model.parameters())}
            if missing:
                raise ValueError(f"Adapter weights match no model parameter: {sorted(missing)[:5]}")
            self._model.load_weights(list(weights.items()), strict=False)
            mx.eval(self._model.parameters())

            logger.info(f"Combined {len(self.adapters)} adapters (rank {rank})")

    def _apply_lora_scale(self, scale: float) -> None:
        """Multiply every LoRA layer's trained scale by ``scale``.

        1.0 leaves the adapter as trained, 0.0 is the base model, above 1.0
        strengthens the style.
        """
        count = 0
        for _, module in self._model.named_modules():
            if hasattr(module, "lora_a") and hasattr(module, "scale"):
                module.scale = module.scale * scale
                count += 1
        if not count:
            raise RuntimeError("No LoRA layers found to scale")
        logger.info(f"Applied LoRA scale x{scale} to {count} layers")

    def _build_logit_bias_processor(self):
        """Build a logits processor that adds per-token bias every step.

        Reads self.config.logit_bias (Dict[str, float]) and resolves each key
        to token IDs via the tokenizer. Handles BPE quirks by trying both the
        bare string and a leading-space variant — most subword tokenizers
        encode ";" and " ;" to different token IDs and the model may emit
        either. Strings that encode to more than one token are skipped with
        a warning (biasing a partial multi-token sequence causes artifacts).

        Returns a callable (tokens, logits) -> logits, or None if the bias
        map is empty or no keys resolved cleanly.
        """
        import mlx.core as mx

        bias_map = self.config.logit_bias
        if not bias_map:
            return None

        resolved: Dict[int, float] = {}
        for s, bias in bias_map.items():
            if not isinstance(bias, (int, float)):
                logger.warning(f"logit_bias: non-numeric value for {s!r}, skipped")
                continue
            added = False
            for variant in (s, " " + s):
                try:
                    ids = self._tokenizer.encode(variant, add_special_tokens=False)
                except Exception as e:
                    logger.warning(f"logit_bias: encode failed for {variant!r}: {e}")
                    continue
                if len(ids) == 1:
                    resolved[ids[0]] = float(bias)
                    added = True
            if not added:
                logger.warning(
                    f"logit_bias: {s!r} did not resolve to any single-token "
                    f"variant (bare or leading-space) — skipped"
                )

        if not resolved:
            return None

        vocab_size = self._model.args.vocab_size if hasattr(self._model, "args") else None
        if vocab_size is None:
            max_id = max(resolved.keys())
            vocab_size = max_id + 1

        import numpy as np
        bias_np = np.zeros(vocab_size, dtype=np.float32)
        for tok_id, bias in resolved.items():
            bias_np[tok_id] = bias
        bias_vec = mx.array(bias_np)

        logger.info(
            f"logit_bias: applied to {len(resolved)} token IDs "
            f"({', '.join(f'{s!r}={v:+.2f}' for s, v in bias_map.items())})"
        )

        def processor(tokens, logits):
            return logits + bias_vec

        return processor

    def generate(
        self,
        content: str,
        author: str,
        max_tokens: Optional[int] = None,
        target_words: Optional[int] = None,
        structural_guidance: Optional[str] = None,
        raw_prompt: bool = False,
        temperature: Optional[float] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Generate styled text from content description.

        Args:
            content: What to express (neutral text to restyle).
            author: Author name (used in prompt).
            max_tokens: Override for max tokens (defaults to config).
            target_words: Target word count for output.
            structural_guidance: Formatted structural guidance (rhythm, punctuation hints).
                           Use get_structural_guidance() to generate.
            raw_prompt: If True, use content directly as prompt without formatting.
                       Used when content is already a fully-formed prompt (e.g., persona prompt).
            temperature: Override for sampling temperature (defaults to config).
            instruction: Persona instruction. When given, ``content`` is only the
                input text and the two share one user turn, as in training.
                        Lower values (0.1-0.3) for more deterministic repairs.

        Returns:
            Generated text in the author's style.
        """
        self._ensure_loaded()

        # Build user message - just the content
        user = content

        # Estimate target word count from input if not provided
        input_words = len(user.split())
        if target_words is None:
            target_words = input_words

        # Calculate tokens based on input length
        # Training data has ~1:1 ratio, but we need enough for complete sentences
        # Convert words to tokens (roughly 1.3 tokens per word)
        # Use 2x input to allow for style variation
        auto_max_tokens = max(100, int(input_words * 2.0 * 1.3))

        if instruction is not None:
            prompt = f"{instruction}\n\n{content}\n###"
        elif raw_prompt:
            # Use content directly as prompt (for persona-injected prompts)
            prompt = content
        else:
            # Generate style tag from input to guide output structure
            style_tag = generate_style_tag(user)

            # Format structural guidance (adds newline prefix if present)
            guidance_str = ""
            if structural_guidance:
                guidance_str = "\n\nSTRUCTURAL GUIDANCE:\n" + structural_guidance + "\n"

            # Build prompt matching training data format EXACTLY
            prompt = format_prompt(
                "style_transfer",
                author=author,
                content=user,
                word_count=target_words,
                style_tag=style_tag,
                structural_guidance=guidance_str,
            )

        # Same layout and template as the LlamaFactory training rows.
        if instruction is not None:
            prompt = self.chat_prompt(instruction, content)
        else:
            prompt = self.chat_prompt(*split_legacy_prompt(prompt))

        # Create sampler with temperature, top_p, and min_p
        # min_p filters low-probability nonsense while allowing creative choices
        # Use override temperature if provided (for repairs)
        effective_temp = (
            temperature if temperature is not None else self.config.temperature
        )
        sampler = make_sampler(
            temp=effective_temp,
            top_p=self.config.top_p,
            min_p=self.config.min_p,
        )

        # Create repetition penalty processor
        rep_penalty = make_repetition_penalty(
            penalty=self.config.repetition_penalty,
            context_size=50,
        )

        # Build logit-bias processor on first call, reuse on subsequent calls.
        # Sentinel None = not-yet-built; False = built-but-empty (no bias map).
        if self._logit_bias_processor is None:
            built = self._build_logit_bias_processor()
            self._logit_bias_processor = built if built is not None else False

        logits_processors = [rep_penalty]
        if self._logit_bias_processor:
            logits_processors.append(self._logit_bias_processor)

        # Use provided max_tokens, or auto-calculated limit, or config default
        # Prefer tighter auto-calculated limit to prevent repetition
        if max_tokens:
            tokens_limit = max_tokens
        else:
            tokens_limit = min(auto_max_tokens, self.config.max_tokens)

        # Training rows end in <|im_end|>; base model tokenizers only stop at
        # <|endoftext|>.
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id not in self._tokenizer.eos_token_ids:
            self._tokenizer.eos_token_ids.add(im_end_id)

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=tokens_limit,
            sampler=sampler,
            logits_processors=logits_processors,
        )

        response = response.strip()

        # Skip cleaning if configured (useful for debugging)
        if self.config.skip_cleaning:
            logger.debug("Skipping _clean_response (skip_cleaning=True)")
            return response

        # Clean up the response
        raw_response = response
        response = self._clean_response(response)

        # Log if cleaning removed significant content
        raw_words = len(raw_response.split())
        clean_words = len(response.split())
        if clean_words < raw_words * 0.7:
            # Usually this means repetition was removed (model repeats after ### marker)
            logger.debug(
                f"_clean_response removed {raw_words - clean_words} words ({raw_words} → {clean_words}) - likely repetition"
            )

        return response

    def unload(self) -> None:
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None

        # Clean up temp checkpoint directories if they exist
        if self._temp_dirs:
            import shutil

            for temp_dir in self._temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            self._temp_dirs = []

        logger.info("Model unloaded")
