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
from typing import Optional, List

from ..utils.logging import get_logger
from ..utils.prompts import format_prompt
from .base_generator import BaseStyleGenerator, GenerationConfig, generate_style_tag

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

    @classmethod
    def from_file(cls, path: Path) -> "AdapterMetadata":
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            author=data.get("author", "Unknown"),
            base_model=data.get("base_model", "mlx-community/Qwen3-8B-Base-bf16"),
            lora_rank=data.get("lora_rank", 16),
            lora_alpha=data.get("lora_alpha", 32),
            epochs=data.get("epochs", 3),
            training_examples=data.get("training_examples", 0),
        )


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
        if ':' in spec:
            parts = spec.rsplit(':', 1)
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
            self.adapters = [AdapterSpec(path=adapter_path, scale=1.0, checkpoint=checkpoint)]
            self.adapter_path = adapter_path
            self.checkpoint = checkpoint
        else:
            # No adapters (base model only)
            self.adapters = []
            self.adapter_path = None
            self.checkpoint = None

        # Lazy load model
        self._model = None
        self._tokenizer = None

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
            # Multiple adapters - load base model, then combine adapter weights
            logger.info(f"Loading {len(self.adapters)} adapters:")
            for adapter in self.adapters:
                logger.info(f"  - {adapter.path} (scale={adapter.scale})")

            # Load first adapter to set up LoRA structure
            first_adapter = self.adapters[0]
            first_path = self._get_effective_adapter_path(first_adapter)

            self._model, self._tokenizer = load(
                self.base_model_name,
                adapter_path=first_path,
            )

            # Get first adapter's weights (already loaded, but we need them for combining)
            first_weights = self._load_adapter_weights(first_path)

            # Scale first adapter's weights
            combined_weights = {
                k: v * first_adapter.scale for k, v in first_weights.items()
            }

            # Load and add remaining adapters
            for adapter in self.adapters[1:]:
                adapter_path = self._get_effective_adapter_path(adapter)
                weights = self._load_adapter_weights(adapter_path)

                for k, v in weights.items():
                    if k in combined_weights:
                        # Check shape compatibility
                        if combined_weights[k].shape != v.shape:
                            raise ValueError(
                                f"Cannot blend adapters with different LoRA ranks.\n"
                                f"  Weight '{k}':\n"
                                f"    - {first_adapter.path}: shape {combined_weights[k].shape}\n"
                                f"    - {adapter.path}: shape {v.shape}\n"
                                f"Adapters must have the same rank to be blended. "
                                f"Use adapters trained with matching configurations."
                            )
                        combined_weights[k] = combined_weights[k] + v * adapter.scale
                    else:
                        combined_weights[k] = v * adapter.scale

            # Apply combined weights to model
            self._model.load_weights(list(combined_weights.items()), strict=False)
            mx.eval(self._model.parameters())

            logger.info(f"Combined {len(self.adapters)} adapters")

    def _apply_lora_scale(self, scale: float) -> None:
        """Apply scaling factor to LoRA adapter weights.

        This controls how much the LoRA adapter influences the base model:
        - scale=0.0: Base model only (no LoRA influence)
        - scale=0.5: Half LoRA influence (more base model)
        - scale=1.0: Full LoRA influence (default)
        - scale>1.0: Amplified LoRA influence (stronger style)

        Args:
            scale: Scaling factor for LoRA weights.
        """
        def scale_lora_layers(module, path=""):
            """Recursively find and scale LoRA layers."""
            # Check if this module has LoRA weights
            if hasattr(module, 'lora_a') and hasattr(module, 'lora_b'):
                # Scale the LoRA output by adjusting lora_b (more efficient than scaling both)
                if hasattr(module, 'scale'):
                    # If module has a scale attribute, use it
                    module.scale = scale
                    logger.debug(f"Scaled {path}.scale = {scale}")
                else:
                    # Otherwise, scale lora_b directly
                    module.lora_b = module.lora_b * scale
                    logger.debug(f"Scaled {path}.lora_b by {scale}")

            # Recurse into children
            if hasattr(module, 'children'):
                for name, child in module.children().items():
                    scale_lora_layers(child, f"{path}.{name}" if path else name)
            elif hasattr(module, '__dict__'):
                for name, child in module.__dict__.items():
                    if hasattr(child, 'lora_a') or hasattr(child, 'children'):
                        scale_lora_layers(child, f"{path}.{name}" if path else name)

        try:
            scale_lora_layers(self._model)
            logger.info(f"Applied LoRA scale: {scale}")
        except Exception as e:
            logger.warning(f"Could not apply LoRA scale: {e}")

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

        if raw_prompt:
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

        # Wrap prompt in chat format if tokenizer supports it
        # LLaMA-Factory trained models expect Qwen chat format
        # Training format: instruction (persona+constraints) | input (content) | output
        if hasattr(self._tokenizer, 'apply_chat_template'):
            # Split at "###" - everything before last content block is instruction
            # Format: {instruction}\n\n{content}\n###
            if '###' in prompt:
                # Find the content by looking for the last double-newline before ###
                prompt_no_stop = prompt.rsplit('###', 1)[0].rstrip()
                # Split instruction from content - content is after last \n\n
                parts = prompt_no_stop.rsplit('\n\n', 1)
                if len(parts) == 2:
                    instruction, user_content = parts
                else:
                    instruction = ""
                    user_content = parts[0]
            else:
                # Fallback: use whole prompt as user content
                instruction = ""
                user_content = prompt

            messages = [
                {'role': 'system', 'content': instruction},
                {'role': 'user', 'content': user_content}
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug("Applied chat template to prompt")

        # Create sampler with temperature, top_p, and min_p
        # min_p filters low-probability nonsense while allowing creative choices
        # Use override temperature if provided (for repairs)
        effective_temp = temperature if temperature is not None else self.config.temperature
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

        # Use provided max_tokens, or auto-calculated limit, or config default
        # Prefer tighter auto-calculated limit to prevent repetition
        if max_tokens:
            tokens_limit = max_tokens
        else:
            tokens_limit = min(auto_max_tokens, self.config.max_tokens)

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=tokens_limit,
            sampler=sampler,
            logits_processors=[rep_penalty],
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
            logger.debug(f"_clean_response removed {raw_words - clean_words} words ({raw_words} → {clean_words}) - likely repetition")

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
