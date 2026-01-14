"""PyTorch-based style transfer generator using HuggingFace PEFT.

This module provides style transfer using PyTorch and PEFT adapters,
enabling inference on Linux/CUDA systems as an alternative to MLX.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from ..utils.logging import get_logger
from ..utils.prompts import format_prompt
from .base_generator import BaseStyleGenerator, GenerationConfig, generate_style_tag

logger = get_logger(__name__)


# Check PyTorch/PEFT availability at module level
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    logger.warning("PyTorch/PEFT not available. Install with: pip install torch transformers peft")


@dataclass
class PyTorchAdapterMetadata:
    """Metadata about a PEFT adapter."""

    author: str
    base_model: str
    lora_rank: int = 16
    lora_alpha: int = 32

    @classmethod
    def from_adapter_config(cls, adapter_path: str) -> "PyTorchAdapterMetadata":
        """Load metadata from PEFT adapter_config.json."""
        import json
        config_path = Path(adapter_path) / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                data = json.load(f)
            return cls(
                author="Unknown",
                base_model=data.get("base_model_name_or_path", ""),
                lora_rank=data.get("r", 16),
                lora_alpha=data.get("lora_alpha", 32),
            )
        return cls(author="Unknown", base_model="")


class PyTorchStyleGenerator(BaseStyleGenerator):
    """Style transfer generator using PyTorch and HuggingFace PEFT.

    This provides an alternative backend to MLX for systems with CUDA GPUs
    or for users who want to use PEFT adapters directly without conversion.

    Example usage:
        generator = PyTorchStyleGenerator(
            adapter_path="yogthos/qwen2.5-32b-lovecraft-lora",
            base_model="Qwen/Qwen2.5-32B-Instruct",
            config=GenerationConfig(temperature=0.7),
        )

        output = generator.generate(
            content="The universe is vast.",
            author="H.P. Lovecraft",
        )
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-32B-Instruct",
        config: Optional[GenerationConfig] = None,
        checkpoint: Optional[str] = None,
        device: str = "auto",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        """Initialize the PyTorch generator.

        Args:
            adapter_path: Path to PEFT adapter (local path or HuggingFace repo ID).
            base_model: HuggingFace model ID for base model.
            config: Generation configuration.
            checkpoint: Specific checkpoint subfolder (e.g., "checkpoint-600").
            device: Device to use ("auto", "cuda", "cpu", "mps").
            load_in_4bit: Load base model with 4-bit quantization (requires bitsandbytes).
            load_in_8bit: Load base model with 8-bit quantization (requires bitsandbytes).
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch/PEFT is not available. Install with:\n"
                "pip install torch transformers peft bitsandbytes accelerate"
            )

        super().__init__(config or GenerationConfig())

        self.adapter_path = adapter_path
        self.base_model_name = base_model
        self.checkpoint = checkpoint
        self.device = self._resolve_device(device)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        # Lazy loading
        self._model = None
        self._tokenizer = None
        self.metadata: Optional[PyTorchAdapterMetadata] = None

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _get_effective_adapter_path(self) -> Optional[str]:
        """Get the effective adapter path including checkpoint subfolder."""
        if not self.adapter_path:
            return None

        if self.checkpoint:
            # Check if it's a local path
            local_path = Path(self.adapter_path) / self.checkpoint
            if local_path.exists():
                return str(local_path)
            # For HuggingFace repos, we'll use subfolder parameter in from_pretrained
            return self.adapter_path
        return self.adapter_path

    def _ensure_loaded(self):
        """Load model and adapter on first use."""
        if self._model is not None:
            return

        logger.info(f"Loading base model: {self.base_model_name}")
        logger.info(f"Device: {self.device}, 4-bit: {self.load_in_4bit}, 8-bit: {self.load_in_8bit}")

        # Configure quantization (only works with CUDA)
        quantization_config = None
        if self.load_in_4bit and self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info("Using 4-bit quantization with bitsandbytes")
            except Exception as e:
                logger.warning(f"Could not configure 4-bit quantization: {e}")
                quantization_config = None
        elif self.load_in_8bit and self.device == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization with bitsandbytes")
            except Exception as e:
                logger.warning(f"Could not configure 8-bit quantization: {e}")
                quantization_config = None

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        elif self.device != "cpu":
            model_kwargs["device_map"] = self.device
        else:
            model_kwargs["device_map"] = "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )

        # Load PEFT adapter if specified
        effective_path = self._get_effective_adapter_path()
        if effective_path:
            logger.info(f"Loading PEFT adapter: {effective_path}")

            # Check if using subfolder for checkpoint
            peft_kwargs = {}
            if self.checkpoint and not Path(effective_path).exists():
                # HuggingFace repo with subfolder
                peft_kwargs["subfolder"] = self.checkpoint

            self._model = PeftModel.from_pretrained(
                self._model,
                effective_path,
                **peft_kwargs,
            )

            # Load metadata
            self.metadata = PyTorchAdapterMetadata.from_adapter_config(effective_path)

            # Apply scale if not 1.0
            if self.config.scale != 1.0:
                self._apply_adapter_scale(self.config.scale)

        logger.info("Model loaded successfully")

    def _apply_adapter_scale(self, scale: float):
        """Apply scaling to adapter weights.

        PEFT models have a scaling mechanism built into the LoRA layers.
        """
        try:
            # For PEFT models, we can adjust the scaling factor
            for name, module in self._model.named_modules():
                if hasattr(module, 'scaling'):
                    # PEFT LoRA layers have a scaling dict
                    for adapter_name in module.scaling:
                        original_scale = module.scaling[adapter_name]
                        module.scaling[adapter_name] = original_scale * scale
                        logger.debug(f"Scaled {name}.scaling[{adapter_name}] by {scale}")
            logger.info(f"Applied adapter scale: {scale}")
        except Exception as e:
            logger.warning(f"Could not apply adapter scale: {e}")

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
        """Generate styled text using PyTorch/HuggingFace.

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
        self._ensure_loaded()

        # Build user message
        user = content

        # Estimate target word count from input if not provided
        input_words = len(user.split())
        if target_words is None:
            target_words = input_words

        # Calculate tokens based on input length (same logic as MLX version)
        auto_max_tokens = max(100, int(input_words * 2.0 * 1.3))

        if raw_prompt:
            prompt = content
        else:
            # Generate style tag from input
            style_tag = generate_style_tag(user)

            # Format structural guidance
            guidance_str = ""
            if structural_guidance:
                guidance_str = "\n\nSTRUCTURAL GUIDANCE:\n" + structural_guidance + "\n"

            # Build prompt matching training data format
            prompt = format_prompt(
                "style_transfer",
                author=author,
                content=user,
                word_count=target_words,
                style_tag=style_tag,
                structural_guidance=guidance_str,
            )

        # Apply chat template if tokenizer supports it
        if hasattr(self._tokenizer, 'apply_chat_template'):
            if '###' in prompt:
                prompt_no_stop = prompt.rsplit('###', 1)[0].rstrip()
                parts = prompt_no_stop.rsplit('\n\n', 1)
                if len(parts) == 2:
                    instruction, user_content = parts
                else:
                    instruction = ""
                    user_content = parts[0]
            else:
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

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        # Determine generation parameters
        effective_temp = temperature if temperature is not None else self.config.temperature
        if max_tokens:
            tokens_limit = max_tokens
        else:
            tokens_limit = min(auto_max_tokens, self.config.max_tokens)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=tokens_limit,
                temperature=effective_temp,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode response (skip input tokens)
        input_length = inputs['input_ids'].shape[1]
        response = self._tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        response = response.strip()

        # Skip cleaning if configured
        if self.config.skip_cleaning:
            logger.debug("Skipping _clean_response (skip_cleaning=True)")
            return response

        # Clean up the response (inherited from base class)
        raw_response = response
        response = self._clean_response(response)

        # Log if cleaning removed significant content
        raw_words = len(raw_response.split())
        clean_words = len(response.split())
        if clean_words < raw_words * 0.7:
            logger.debug(f"_clean_response removed {raw_words - clean_words} words")

        return response

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Clear CUDA cache
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")
