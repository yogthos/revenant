"""Generation module for LoRA-based style transfer.

The primary pipeline uses LoRA adapters for fast, consistent style transfer:
- LoRAStyleGenerator: Core generation using MLX LoRA adapters
- PyTorchStyleGenerator: Alternative generation using PyTorch/PEFT
- StyleTransfer: High-level pipeline with semantic graph validation
- DocumentContext: Document-level context for improved coherence
"""

from .base_generator import (
    BaseStyleGenerator,
    GenerationConfig,
    generate_style_tag,
)
from .lora_generator import (
    LoRAStyleGenerator,
    AdapterMetadata,
    AdapterSpec,
)
from .factory import (
    create_style_generator,
    detect_best_backend,
    list_available_backends,
)
from .transfer import (
    StyleTransfer,
    TransferConfig,
    TransferStats,
)
from .document_context import (
    DocumentContext,
    DocumentContextExtractor,
    extract_document_context,
)

# Conditionally export PyTorch generator
try:
    from .pytorch_generator import PyTorchStyleGenerator
    _pytorch_available = True
except ImportError:
    PyTorchStyleGenerator = None  # type: ignore
    _pytorch_available = False

__all__ = [
    # Base generator
    "BaseStyleGenerator",
    "GenerationConfig",
    "generate_style_tag",
    # MLX LoRA generation
    "LoRAStyleGenerator",
    "AdapterMetadata",
    "AdapterSpec",
    # Factory
    "create_style_generator",
    "detect_best_backend",
    "list_available_backends",
    # Style transfer pipeline
    "StyleTransfer",
    "TransferConfig",
    "TransferStats",
    # Document context
    "DocumentContext",
    "DocumentContextExtractor",
    "extract_document_context",
]

# Add PyTorch generator to exports if available
if _pytorch_available:
    __all__.append("PyTorchStyleGenerator")
