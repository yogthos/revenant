"""Style transfer pipeline using LoRA with RTT neutralization.

This module provides a style transfer pipeline that uses LoRA-adapted models
for consistent style transfer with semantic fidelity validation.

Pipeline:
1. RTT neutralization (English → Mandarin HSK5 → Plain English) to strip style
2. Pass neutralized text to LoRA for style application
3. Validate semantic fidelity via DeepSeek (single-step replacement for
   NLI verification, repair loops, grammar correction, and repetition reduction)
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import List, Optional, Callable, Tuple, TYPE_CHECKING
import re
import time

if TYPE_CHECKING:
    from ..services import Services

from .lora_generator import AdapterSpec
from .base_generator import GenerationConfig
from .factory import create_style_generator
from ..config import ModelConfig, get_adapter_config, get_fused_model_config
from ..utils.nlp import (
    split_into_paragraphs,
    split_into_sentences,
    is_heading,
)
from ..utils.prompts import load_prompt
from ..utils.logging import get_logger

# Optional Structural RAG import
try:
    from ..rag import StructuralRAG, get_structural_rag

    STRUCTURAL_RAG_AVAILABLE = True
except ImportError:
    STRUCTURAL_RAG_AVAILABLE = False
    StructuralRAG = None

# Optional Structural Grafter import
try:
    from ..rag import StructuralGrafter, get_structural_grafter

    STRUCTURAL_GRAFTER_AVAILABLE = True
except ImportError:
    STRUCTURAL_GRAFTER_AVAILABLE = False
    StructuralGrafter = None

# Persona system for subjective style transfer
try:
    from ..persona import build_persona_instruction

    PERSONA_AVAILABLE = True
except ImportError:
    PERSONA_AVAILABLE = False

logger = get_logger(__name__)


def _with_self_services(method):
    """Run a StyleTransfer method under its own self.services container.

    Nested helpers (split_into_sentences via get_nlp, is_heading, structural
    analyzers, …) resolve through get_default_services(). Wrapping the
    public entry points ensures they see the injected container instead of
    whatever process-wide default another thread or test set up.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        from ..services import default_services
        with default_services(self.services):
            return method(self, *args, **kwargs)

    return wrapper


def _apply_fused_model_overrides(
    transfer_config: "TransferConfig", fused_cfg: ModelConfig
) -> None:
    """Apply per-fused-model overrides to a TransferConfig in place.

    Mirrors the adapter override block: CLI flags win over config, so
    expand_for_texture only overrides when expand_for_texture_explicit is False.
    """
    if (
        fused_cfg.expand_for_texture is not None
        and not transfer_config.expand_for_texture_explicit
    ):
        transfer_config.expand_for_texture = fused_cfg.expand_for_texture
        logger.info(
            f"Using fused-model expand_for_texture={fused_cfg.expand_for_texture}"
        )

    if fused_cfg.perspective is not None:
        transfer_config.perspective = fused_cfg.perspective
        logger.info(f"Using fused-model perspective={fused_cfg.perspective}")

    if (
        fused_cfg.verify_entailment is not None
        and not transfer_config.verify_semantic_fidelity_explicit
    ):
        transfer_config.verify_semantic_fidelity = fused_cfg.verify_entailment
        logger.info(
            f"Using fused-model verify_semantic_fidelity={fused_cfg.verify_entailment}"
        )

    if fused_cfg.merge_paragraphs is not None:
        transfer_config.merge_paragraphs = fused_cfg.merge_paragraphs
        logger.info(
            f"Using fused-model merge_paragraphs={fused_cfg.merge_paragraphs}"
        )

    if fused_cfg.use_structural_rag is not None:
        transfer_config.use_structural_rag = fused_cfg.use_structural_rag
        logger.info(
            f"Using fused-model use_structural_rag={fused_cfg.use_structural_rag}"
        )


@dataclass
class TransferConfig:
    """Configuration for style transfer."""

    # Temperature override from CLI (None = use per-adapter/model config).
    # Other sampling params (max_tokens, top_p, min_p, repetition_penalty)
    # live on GenerationConfig and are driven by the model entry in config.json.
    temperature: Optional[float] = None

    # Semantic fidelity validation (single DeepSeek call replaces NLI + repair + grammar + repetition)
    verify_semantic_fidelity: bool = True

    # Content handling
    pass_headings_unchanged: bool = True  # Don't transform headings
    min_paragraph_words: int = 10  # Skip very short paragraphs

    # Input format (uses graph-based description matching training format)

    # Length control settings
    max_expansion_ratio: float = 2.5  # Max output/input word ratio before warning
    target_expansion_ratio: float = (
        1.25  # Median output/input word ratio in the training rows
    )
    expand_for_texture: bool = (
        False  # Add stronger expansion prompt for texture/flourishes
    )
    expand_for_texture_explicit: bool = (
        False  # True when set by CLI flag (takes priority over adapter config)
    )
    verify_semantic_fidelity_explicit: bool = (
        False  # True when set by --no-verify (takes priority over adapter config)
    )

    # Paragraph merging — merge short paragraphs to reach minimum word count
    # LoRAs produce better rhythm/burstiness with longer input blocks (~200+ words)
    merge_paragraphs: Optional[int] = (
        None  # None = no merging, int = min words per block
    )

    # Neutralization settings
    skip_neutralization: bool = (
        False  # If True, skip RTT and use original text as input
    )

    # Perspective settings
    perspective: str = (
        "preserve"  # preserve, first_person_singular, first_person_plural, third_person
    )

    # Structural RAG settings
    use_structural_rag: bool = True  # Enable Structural RAG for rhythm/syntax guidance
    use_structural_grafting: bool = (
        True  # Enable Structural Grafting for argument skeletons
    )
    rag_sample_size: int = (
        300  # Number of corpus chunks to sample for rhythm pattern analysis
    )

    # Persona settings (subjective voice to defeat AI detection)
    use_persona: bool = True  # Enable persona-based prompting
    apply_input_perturbation: bool = (
        True  # Apply 8% noise to match training distribution
    )


@dataclass
class TransferStats:
    """Statistics from a transfer operation."""

    paragraphs_processed: int = 0
    total_time_seconds: float = 0.0
    avg_time_per_paragraph: float = 0.0
    entailment_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paragraphs_processed": self.paragraphs_processed,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_paragraph": round(self.avg_time_per_paragraph, 2),
            "avg_entailment_score": round(
                sum(self.entailment_scores) / len(self.entailment_scores), 3
            )
            if self.entailment_scores
            else 0.0,
        }


class StyleTransfer:
    """Style transfer using LoRA with RTT neutralization.

    This is the main entry point for style transfer. Pipeline:

    1. RTT neutralize input (English → Mandarin → English)
    2. Pass neutralized text to LoRA for style application
    3. Validate semantic fidelity via DeepSeek

    Example usage:
        transfer = StyleTransfer(
            adapter_path="lora_adapters/sagan",
            author_name="Carl Sagan",
            critic_provider=deepseek_provider,
        )

        result, stats = transfer.transfer_document(input_text)
        print(result)
    """

    def __init__(
        self,
        adapter_path: Optional[str],
        author_name: str,
        critic_provider,
        config: Optional[TransferConfig] = None,
        verify_fn: Optional[Callable[[str, str], float]] = None,
        checkpoint: Optional[str] = None,
        adapters: Optional[List[AdapterSpec]] = None,
        fused_models: Optional[List[str]] = None,
        services: Optional["Services"] = None,
    ):
        """Initialize the fast transfer pipeline.

        Args:
            adapter_path: Path to LoRA adapter directory, or None for base model.
            author_name: Author name for prompts.
            critic_provider: LLM provider for critique/repair (e.g., DeepSeek).
            config: Transfer configuration.
            verify_fn: Optional verification function (original, output) -> score.
            checkpoint: Specific checkpoint file to use (e.g., "0000600_adapters.safetensors").
            adapters: List of AdapterSpec for multiple adapters. If provided, adapter_path is ignored.
            fused_models: List of fused model paths to use directly (no adapter needed).
            services: Optional Services container for dependency injection. If
                None, the process-wide default from get_default_services() is used.
        """
        self.config = config or TransferConfig()
        self.author = author_name
        self.verify_fn = verify_fn

        if services is None:
            from ..services import get_default_services
            services = get_default_services()
        self.services = services

        # Convert string provider name to actual LLMProvider object if needed
        if isinstance(critic_provider, str):
            from ..llm.provider import create_critic_provider
            from ..config import load_config

            app_config = load_config()
            self.critic_provider = create_critic_provider(app_config.llm)
        else:
            self.critic_provider = critic_provider

        # Log key config settings
        logger.info(
            f"StyleTransfer config: expand_for_texture={self.config.expand_for_texture}, "
            f"target_expansion_ratio={self.config.target_expansion_ratio}"
        )

        fused_models = fused_models or []

        if fused_models:
            primary_adapter_path = fused_models[0]
        elif adapters:
            primary_adapter_path = adapters[0].path
        else:
            primary_adapter_path = adapter_path
        # adapter_path on the instance is consumed by the persona prompt builder
        # to look up the worldview — it must be set for both adapters and fused
        # models so _get_worldview_filename can locate the config entry.
        self.adapter_path = primary_adapter_path

        # Validate persona file at startup so a typo in config.worldview fails
        # fast instead of aborting mid-document on the first paragraph.
        if self.config.use_persona and PERSONA_AVAILABLE:
            from ..persona.prompt_builder import (
                _get_worldview_filename,
                _load_persona_file,
            )
            _load_persona_file(_get_worldview_filename(self.adapter_path))

        if fused_models:
            fused_cfg = get_fused_model_config(primary_adapter_path)
            gen_config = GenerationConfig.from_fused_model(fused_cfg)
            if self.config.temperature is not None:
                gen_config.temperature = self.config.temperature
            gen_config.skip_cleaning = False

            _apply_fused_model_overrides(self.config, fused_cfg)

            self.generator = create_style_generator(
                adapter_path=None,
                config=gen_config,
                fused_models=fused_models,
                backend=fused_cfg.backend,
                device=fused_cfg.device,
                load_in_4bit=fused_cfg.load_in_4bit,
                load_in_8bit=fused_cfg.load_in_8bit,
            )
        else:
            gen_config = GenerationConfig.from_config(primary_adapter_path)
            if self.config.temperature is not None:
                gen_config.temperature = self.config.temperature
            gen_config.skip_cleaning = False

            adapter_cfg = get_adapter_config(primary_adapter_path)

            if (
                adapter_cfg.expand_for_texture is not None
                and not self.config.expand_for_texture_explicit
            ):
                self.config.expand_for_texture = adapter_cfg.expand_for_texture
                logger.info(
                    f"Using adapter-specific expand_for_texture={adapter_cfg.expand_for_texture}"
                )

            if adapter_cfg.perspective is not None:
                self.config.perspective = adapter_cfg.perspective
                logger.info(
                    f"Using adapter-specific perspective={adapter_cfg.perspective}"
                )

            if (
                adapter_cfg.verify_entailment is not None
                and not self.config.verify_semantic_fidelity_explicit
            ):
                self.config.verify_semantic_fidelity = adapter_cfg.verify_entailment
                logger.info(
                    f"Using adapter-specific verify_semantic_fidelity={adapter_cfg.verify_entailment}"
                )

            if adapter_cfg.merge_paragraphs is not None:
                self.config.merge_paragraphs = adapter_cfg.merge_paragraphs
                logger.info(
                    f"Using adapter-specific merge_paragraphs={adapter_cfg.merge_paragraphs}"
                )

            if adapter_cfg.use_structural_rag is not None:
                self.config.use_structural_rag = adapter_cfg.use_structural_rag
                logger.info(
                    f"Using adapter-specific use_structural_rag={adapter_cfg.use_structural_rag}"
                )

            self.generator = create_style_generator(
                adapter_path=adapter_cfg.hf_adapter_path or adapter_path,
                config=gen_config,
                checkpoint=checkpoint,
                adapters=adapters,
                backend=adapter_cfg.backend,
                device=adapter_cfg.device,
                load_in_4bit=adapter_cfg.load_in_4bit,
                load_in_8bit=adapter_cfg.load_in_8bit,
            )

        # Initialize RTT neutralizer (local MLX model)
        self._rtt_neutralizer = None

        # Push self.services onto the thread-local default stack while we
        # build collaborators so their module-level helpers
        # (get_structural_analyzer, get_indexer, get_nlp, …) resolve against
        # THIS container instead of the process-wide default. Without this
        # wrapper, self.services is stored but never consulted by nested
        # constructors — DI becomes theatrical.
        from ..services import default_services
        with default_services(self.services):
            # Structural RAG for rhythm/syntax guidance
            self.structural_rag: Optional[StructuralRAG] = None
            if self.config.use_structural_rag:
                if STRUCTURAL_RAG_AVAILABLE:
                    self.structural_rag = get_structural_rag(self.author)
                    loaded = self.structural_rag.load_patterns(
                        sample_size=self.config.rag_sample_size
                    )
                    if loaded > 0:
                        logger.info(
                            f"Structural RAG loaded {loaded} rhythm patterns for {self.author}"
                        )
                    else:
                        logger.warning(f"No structural patterns found for {self.author}")
                        self.structural_rag = None
                else:
                    logger.warning("Structural RAG not available (missing dependencies)")
                    self.config.use_structural_rag = False

            # Structural Grafter for argument skeletons
            self.structural_grafter: Optional[StructuralGrafter] = None
            if self.config.use_structural_grafting:
                if STRUCTURAL_GRAFTER_AVAILABLE:
                    self.structural_grafter = get_structural_grafter(
                        self.author, critic_provider
                    )
                    logger.info(f"Structural Grafter initialized for {self.author}")
                else:
                    logger.warning(
                        "Structural Grafter not available (missing dependencies)"
                    )
                    self.config.use_structural_grafting = False

    def _rtt_neutralize(self, text: str, max_retries: int = 2) -> Optional[str]:
        """Round-Trip Translation neutralization via Mandarin pivot.

        This matches the training data generation process:
        Step 1 (Scrub): English → Mandarin (HSK 5 vocabulary)
        Step 2 (Rinse): Mandarin → Plain English

        Uses provider from config.json under llm.provider.rtt.
        Options: 'mlx' (local), 'deepseek' (API).

        Args:
            text: Input text to neutralize.
            max_retries: Number of retry attempts.

        Returns:
            Neutralized text, or None if failed.
        """
        # Lazy-load the RTT neutralizer using factory function
        if self._rtt_neutralizer is None:
            try:
                from ..llm.mlx_provider import create_rtt_neutralizer

                self._rtt_neutralizer = create_rtt_neutralizer()
                logger.debug(f"RTT neutralizer: {type(self._rtt_neutralizer).__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize RTT neutralizer: {e}")
                return None

        # Training neutralized with monotone flattening; match it.
        return self._rtt_neutralizer.neutralize(text, max_retries=max_retries, monotone=True)

    def _expand_with_texture(self, text: str) -> str:
        """Expand text with texture using the critic model.

        Adds asides, observations, parenthetical thoughts, and sensory details
        to enrich flat prose before style transfer.

        Args:
            text: Input text to expand.

        Returns:
            Expanded text with added texture, or original text if expansion fails.
        """
        try:
            system_prompt = load_prompt("expand_texture")
            user_prompt = text

            response = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Some creativity for texture
                max_tokens=len(text.split()) * 3,  # Allow ~2x expansion headroom
            )

            input_words = len(text.split())
            output_words = len(response.split()) if response else 0
            logger.info(
                f"TEXTURE EXPANSION result: {input_words} → {output_words} words"
            )

            if response and output_words > input_words:
                expansion = output_words / input_words
                logger.info(f"TEXTURE EXPANSION: expanded by {expansion:.0%}")
                return response.strip()
            else:
                logger.warning(
                    f"Texture expansion returned shorter/equal text ({output_words} vs {input_words}), using original"
                )
                return text

        except Exception as e:
            logger.warning(f"Texture expansion failed: {e}")
            return text

    def _narrativize(self, text: str) -> str:
        """Convert impersonal exposition to first-person narrative.

        CRITICAL FOR LORA QUALITY:
        The LoRA was trained on first-person narrative inputs ("I saw", "I found",
        "I discovered"). But RTT neutralization produces impersonal exposition
        ("We trace", "One observes", "It is known that").

        This step bridges that gap by converting input to match training format:
        - "We now trace the forces..." → "I have traced the forces..."
        - "One must understand..." → "I came to understand..."
        - "It is observed that..." → "I observed..."

        Args:
            text: Impersonal exposition text.

        Returns:
            First-person narrative version, or original text if conversion fails.
        """
        try:
            system_prompt = load_prompt("narrativize")
            user_prompt = text

            response = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5,  # Some variation but controlled
                max_tokens=len(text.split()) * 2,  # Allow for slight expansion
            )

            if response and response.strip():
                input_words = len(text.split())
                output_words = len(response.split())
                logger.info(
                    f"NARRATIVIZE: {input_words} → {output_words} words (converted to first-person)"
                )
                return response.strip()
            else:
                logger.warning("Narrativization returned empty, using original")
                return text

        except Exception as e:
            logger.warning(f"Narrativization failed: {e}")
            return text

    def _convert_to_perspective(self, text: str, target_perspective: str) -> str:
        """Convert text to target perspective BEFORE RTT neutralization.

        CRITICAL: This must happen BEFORE RTT because the LoRA was trained on
        perspective-varied text that went through RTT. The training pairs are:
            neutral(third_person) → styled(third_person)

        So the perspective is embedded in the text BEFORE RTT, and the LoRA
        preserves it during styling.

        Args:
            text: Input text in any perspective.
            target_perspective: Target perspective from config.

        Returns:
            Text converted to target perspective.
        """
        # "preserve" means don't convert - keep original perspective
        if target_perspective == "preserve":
            return text

        # "first_person_singular" uses the existing narrativize prompt
        if target_perspective == "first_person_singular":
            return self._narrativize(text)

        try:
            # Build the perspective description
            perspective_descriptions = {
                "first_person_plural": "first_person_plural (use: we, us, our, ours)",
                "third_person": "third_person (use: the observer, they, one)",
                "author_voice_third_person": "author_voice_third_person (impersonal exposition: one observes, it is known, passive voice)",
            }
            perspective_desc = perspective_descriptions.get(
                target_perspective, target_perspective
            )

            system_prompt = load_prompt("convert_perspective").format(
                target_perspective=perspective_desc
            )
            user_prompt = text

            response = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for precise conversion
                max_tokens=len(text.split()) * 2,
            )

            if response and response.strip():
                input_words = len(text.split())
                output_words = len(response.split())
                logger.info(
                    f"PERSPECTIVE CONVERSION: {input_words} → {output_words} words "
                    f"(converted to {target_perspective})"
                )
                return response.strip()
            else:
                logger.warning("Perspective conversion returned empty, using original")
                return text

        except Exception as e:
            logger.warning(f"Perspective conversion failed: {e}")
            return text

    @_with_self_services
    def transfer_paragraph(
        self,
        paragraph: str,
        previous: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Transfer a single paragraph with graph-based validation.

        Pipeline:
        1. Build source semantic graph (ground truth)
        2. Generate neutral prose from graph (deterministic, all propositions)
        3. Pass neutral prose to LoRA writer for styling
        4. Validate styled output against source graph
        5. Repair any missing propositions
        6. Final LoRA restyle pass

        Args:
            paragraph: Source paragraph.
            previous: Previous output paragraph for continuity.

        Returns:
            Tuple of (styled_paragraph, entailment_score).
        """
        from ..validation.reference_tracker import (
            extract_references,
            reinject_references,
        )

        # Skip very short paragraphs
        if len(paragraph.split()) < self.config.min_paragraph_words:
            logger.debug(f"Skipping short paragraph: {paragraph[:50]}...")
            return paragraph, 1.0

        # ========================================
        # STEP 0: Extract and preserve references [^N]
        # ========================================
        # References are stripped before processing and reinjected at the end
        paragraph_clean, ref_map = extract_references(paragraph)

        word_count = len(paragraph_clean.split())
        logger.debug(f"Translating paragraph: {word_count} words")

        # Save original for semantic verification (before any expansion)
        original_for_verification = paragraph_clean

        # ========================================
        # STEP 0.5: Texture expansion (optional)
        # ========================================
        # If enabled, use the critic model to add asides, observations, and
        # texture before RTT neutralization. This enriches flat prose.
        if self.config.expand_for_texture:
            logger.info(
                f"TEXTURE EXPANSION: Starting expansion for {len(paragraph_clean.split())} words"
            )
            paragraph_clean = self._expand_with_texture(paragraph_clean)
            word_count = len(
                paragraph_clean.split()
            )  # Update word count after expansion
            logger.info(f"TEXTURE EXPANSION: Complete, now {word_count} words")

        # ========================================
        # STEP 0.7: Convert to target perspective BEFORE RTT
        # ========================================
        # CRITICAL ORDERING: Perspective conversion must happen BEFORE RTT because:
        # 1. Training used RTT on perspective-varied text (first-person, third-person, impersonal)
        # 2. The LoRA preserves perspective through RTT → styled output
        # 3. Training pairs: neutral(perspective_text) → styled(perspective_text)
        #
        # Correct flow: input → convert_to_perspective → RTT → neutral_in_perspective → LoRA → styled_in_perspective
        #
        # For backward compatibility:
        # - "preserve" = keep input perspective (no conversion)
        # - "first_person_singular" = convert to first person (legacy narrativize behavior)
        # - Other perspectives = convert to that perspective
        if self.config.perspective != "preserve":
            pre_perspective_words = len(paragraph_clean.split())
            paragraph_clean = self._convert_to_perspective(
                paragraph_clean, self.config.perspective
            )
            post_perspective_words = len(paragraph_clean.split())
            word_count = post_perspective_words  # Update word count for LoRA target
            logger.info(
                f"PERSPECTIVE: {pre_perspective_words} → {post_perspective_words} words (→ {self.config.perspective})"
            )

        # ========================================
        # STEP 1: RTT Neutralization (match training format)
        # ========================================
        # Training used Round-Trip Translation via Mandarin to neutralize text
        # We must use the same process during inference for the LoRA to work
        # Note: Use paragraph_clean (references stripped) for processing
        if self.config.skip_neutralization:
            # Skip RTT - use cleaned text directly
            content_for_generation = paragraph_clean
            logger.debug("RTT neutralization skipped (skip_neutralization=true)")
        else:
            logger.info(
                f"RTT: Starting neutralization for {len(paragraph_clean.split())} words"
            )
            content_for_generation = self._rtt_neutralize(paragraph_clean)
            if not content_for_generation:
                # Fall back to cleaned text instead of crashing
                logger.warning(
                    f"RTT neutralization failed for paragraph: {paragraph_clean[:50]}... "
                    "Falling back to original text. "
                    "Check config.json llm.provider.rtt setting."
                )
                content_for_generation = paragraph_clean
            else:
                rtt_input_words = len(paragraph_clean.split())
                rtt_output_words = len(content_for_generation.split())
                compression_ratio = (
                    rtt_output_words / rtt_input_words if rtt_input_words > 0 else 1.0
                )
                word_count = rtt_output_words  # Update word count for LoRA target
                logger.info(
                    f"RTT: {rtt_input_words} → {rtt_output_words} words ({compression_ratio:.0%})"
                )

        # ========================================
        # STEP 1.5: Apply input perturbation to match training distribution
        # ========================================
        # Training data used 8% perturbation (typos, word drops, synonym swaps)
        # This forces the model to creatively reconstruct, not just restyle
        if self.config.apply_input_perturbation:
            from ..utils.perturbation import perturb_text

            pre_perturb_words = len(content_for_generation.split())
            content_for_generation = perturb_text(
                content_for_generation,
                perturbation_rate=0.08,
            )
            post_perturb_words = len(content_for_generation.split())
            logger.info(
                f"PERTURBATION: {pre_perturb_words} → {post_perturb_words} words (8% noise)"
            )

        # ========================================
        # STEP 2: Pass to LoRA for style transformation
        # ========================================
        target_words = int(word_count * self.config.target_expansion_ratio)
        logger.info(
            f"LORA: content_for_generation={len(content_for_generation.split())} words, target={target_words} words"
        )
        # Token limit needs to be generous to avoid truncation mid-sentence
        # Typically ~1.5 tokens per word, plus some margin for style variation
        # Use 2.5x target words to ensure complete sentences
        max_tokens = max(150, int(target_words * 2.5))

        # Structural guidance from the AUTHOR'S CORPUS (not source text):
        # rhythm patterns and the skeleton of the most similar paragraph.
        # Training rows get the same guidance (filter_training_data.PersonaBuilder).
        structural_guidance = None
        if self.structural_rag:
            structural_guidance = self.structural_rag.get_guidance(paragraph).format_for_prompt()
            logger.debug(f"Using author structural guidance: {structural_guidance[:100]}...")

        grafting_guidance = None
        if self.structural_grafter:
            grafting_guidance = self.structural_grafter.get_grafting_guidance(paragraph)
            if grafting_guidance:
                logger.debug(f"Using grafting skeleton: {grafting_guidance.skeleton.format_for_prompt()}")

        # The generator lays the instruction and the input out the way the
        # adapter's training rows did (see base_generator.render_chat_prompt).
        instruction = None
        if self.config.use_persona and PERSONA_AVAILABLE:
            instruction = build_persona_instruction(
                content=content_for_generation,
                structural_guidance=structural_guidance,
                grafting_guidance=grafting_guidance,
                target_words=target_words,  # Pass word count to match training format
                adapter_path=self.adapter_path,
            )
            structural_guidance = None  # Already included in the instruction
            logger.debug(f"Using persona prompt (target={target_words} words)")

        output = self.generator.generate(
            content=content_for_generation,
            author=self.author,
            max_tokens=max_tokens,
            target_words=target_words,
            structural_guidance=structural_guidance,
            raw_prompt=instruction is not None,
            instruction=instruction,
        )
        lora_output_words = len(output.split())
        logger.info(
            f"LORA OUTPUT: {lora_output_words} words (target was {target_words})"
        )

        # Check if LoRA output matches input (indicates no transformation)
        if output.strip() == content_for_generation.strip():
            logger.warning(
                "LoRA output identical to input - no transformation occurred"
            )

        # Check for memorization (output has no semantic overlap with input)
        output_overlap = self._check_content_overlap(content_for_generation, output)
        if output_overlap < 0.1:
            logger.warning(
                f"Possible memorized output detected (only {output_overlap:.0%} content overlap). "
                "Try lowering lora_scale in config.json or using an earlier checkpoint."
            )

        # Track expansion at LoRA stage
        lora_words = len(output.split())
        source_words = len(paragraph_clean.split())
        if lora_words > source_words * self.config.max_expansion_ratio:
            logger.warning(
                f"LoRA over-expanded: {lora_words} words vs {source_words} source ({lora_words / source_words:.0%})"
            )

        # ========================================
        # STEP 3: Semantic fidelity validation (if enabled)
        # ========================================
        # Single DeepSeek call replaces NLI verification, repair loop,
        # grammar correction, and repetition reduction
        if self.config.verify_semantic_fidelity:
            from ..validation.semantic_fidelity import validate_semantic_fidelity

            logger.info(f"SEMANTIC FIDELITY: validating {len(output.split())} words")
            fidelity = validate_semantic_fidelity(
                original=original_for_verification,
                restyled=output,
                critic_provider=self.critic_provider,
            )
            if fidelity.was_modified:
                logger.info(f"SEMANTIC FIDELITY: {len(fidelity.changes)} fixes applied")
            output = fidelity.corrected

        # Ensure output ends with complete sentence
        output = self._ensure_complete_ending(output)

        # ========================================
        # STEP 4: Reinject references [^N]
        # ========================================
        if ref_map.has_references():
            output, dropped_refs = reinject_references(output, ref_map)
            if dropped_refs:
                logger.warning(
                    f"Lost {len(dropped_refs)} references during style transfer: {dropped_refs}"
                )
            logger.debug(f"Reinjected {len(ref_map.references)} references")

        # Score via verify_fn if configured (for stats tracking)
        score = 1.0
        if self.verify_fn:
            score = self.verify_fn(original_for_verification, output)

        logger.info(f"FINAL OUTPUT: {len(output.split())} words")
        return output, score

    def _check_content_overlap(self, input_text: str, output_text: str) -> float:
        """Check content word overlap between input and output.

        Returns ratio of input content words found in output.
        Low overlap suggests memorized/hallucinated output.
        """
        nlp = self.services.nlp

        def get_content_words(text: str) -> set:
            """Extract lemmatized content words using spaCy."""
            doc = nlp(text)
            words = set()
            for token in doc:
                # Skip stopwords, punctuation, and short words
                if not token.is_stop and not token.is_punct and len(token.lemma_) >= 4:
                    words.add(token.lemma_.lower())
            return words

        input_words = get_content_words(input_text)
        output_words = get_content_words(output_text)

        if not input_words:
            return 1.0  # No content words to check

        overlap = len(input_words & output_words)
        return overlap / len(input_words)

    def _clean_punctuation_artifacts(self, text: str) -> str:
        """Clean up punctuation artifacts from LoRA output and post-processing.

        Fixes common issues like:
        - "—," or ",—" (em-dash combined with comma)
        - ".—" or "—." (em-dash combined with period)
        - Double punctuation
        """
        # Fix em-dash + punctuation combinations
        text = re.sub(r"—\s*,", ",", text)  # "—," -> ","
        text = re.sub(r",\s*—", ",", text)  # ",—" -> ","
        text = re.sub(r"—\s*\.", ".", text)  # "—." -> "."
        text = re.sub(r"\.\s*—", ".", text)  # ".—" -> "."
        text = re.sub(r"—\s*;", ";", text)  # "—;" -> ";"
        text = re.sub(r";\s*—", ";", text)  # ";—" -> ";"
        text = re.sub(r"—\s*:", ":", text)  # "—:" -> ":"
        text = re.sub(r":\s*—", ":", text)  # ":—" -> ":"

        # Fix double punctuation
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"\.\s*\.", ".", text)
        text = re.sub(r";\s*;", ";", text)
        text = re.sub(r":\s*:", ":", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # No space before
        # Space after punctuation, but not between single uppercase letters (abbreviations like U.S.)
        text = re.sub(r"([.,;:!?])(?!(?<=[A-Z]\.)[A-Z])([A-Za-z])", r"\1 \2", text)

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _ensure_complete_ending(self, text: str) -> str:
        """Ensure text ends with a complete sentence.

        If text ends mid-sentence, remove the incomplete part.
        """
        # First clean punctuation artifacts
        text = self._clean_punctuation_artifacts(text)

        text = text.strip()
        if not text:
            return text

        # If already ends with sentence terminator, we're good
        if text[-1] in ".!?":
            return text

        # Find the last complete sentence
        sentences = split_into_sentences(text)
        if not sentences:
            return text

        # Check if last sentence is complete (ends with punctuation)
        complete_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent[-1] in ".!?":
                complete_sentences.append(sent)
            elif sent and len(sent) > 20:
                # Long fragment - try to salvage by adding period
                # Only if it looks like a complete thought
                words = sent.split()
                if len(words) >= 5:
                    complete_sentences.append(sent + ".")
                    logger.warning(
                        f"Added period to incomplete sentence: ...{sent[-30:]}"
                    )

        if complete_sentences:
            return " ".join(complete_sentences)

        # Fallback: add period to entire text
        return text + "."

    @_with_self_services
    def transfer_document(
        self,
        text: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_paragraph: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[str, TransferStats]:
        """Transfer an entire document.

        Args:
            text: Source document text.
            on_progress: Optional callback (current, total, status).
            on_paragraph: Optional callback (index, paragraph) called after each paragraph is complete.

        Returns:
            Tuple of (styled_document, statistics).
        """
        # Track state for partial results on interrupt
        self._transfer_start_time = time.time()
        self._transfer_outputs = []
        self._transfer_stats = TransferStats()

        # Split into paragraphs
        paragraphs = split_into_paragraphs(text)

        if not paragraphs:
            logger.warning("No content paragraphs found")
            return text, self._transfer_stats

        # Merge short paragraphs into larger blocks for better LoRA rhythm.
        # LoRAs produce more natural sentence variation with 200+ word inputs.
        if self.config.merge_paragraphs:
            min_words = self.config.merge_paragraphs
            merged = []
            current_block = []
            current_words = 0

            for para in paragraphs:
                para_words = len(para.split())
                # Always keep headings separate
                if (
                    self.config.pass_headings_unchanged
                    and len(para.strip().split("\n")) == 1
                    and is_heading(para.strip())
                ):
                    if current_block:
                        merged.append("\n\n".join(current_block))
                        current_block = []
                        current_words = 0
                    merged.append(para)
                    continue

                current_block.append(para)
                current_words += para_words

                if current_words >= min_words:
                    merged.append("\n\n".join(current_block))
                    current_block = []
                    current_words = 0

            # Flush remaining — merge with last block if too small
            if current_block:
                if (
                    merged
                    and current_words < min_words // 2
                    and not is_heading(merged[-1].strip())
                ):
                    merged[-1] = merged[-1] + "\n\n" + "\n\n".join(current_block)
                else:
                    merged.append("\n\n".join(current_block))

            logger.info(
                f"Merged {len(paragraphs)} paragraphs into {len(merged)} blocks (min_words={min_words})"
            )
            paragraphs = merged

        logger.info(f"Transferring {len(paragraphs)} paragraphs")

        previous = None

        for i, para in enumerate(paragraphs):
            if on_progress:
                on_progress(i + 1, len(paragraphs), f"Processing paragraph {i + 1}")

            para_start = time.time()

            # Check if paragraph is a heading - pass through unchanged
            para_lines = para.strip().split("\n")
            is_heading_para = (
                self.config.pass_headings_unchanged
                and len(para_lines) == 1
                and is_heading(para_lines[0])
            )

            if is_heading_para:
                logger.debug(f"Passing heading unchanged: {para[:50]}...")
                output = para
                score = 1.0
            else:
                output, score = self.transfer_paragraph(para, previous)

            para_time = time.time() - para_start
            logger.debug(f"Paragraph {i + 1}: {para_time:.1f}s, score={score:.2f}")

            self._transfer_outputs.append(output)
            previous = output

            self._transfer_stats.paragraphs_processed += 1
            self._transfer_stats.entailment_scores.append(score)

            # Notify callback with completed paragraph
            if on_paragraph:
                on_paragraph(i, output)

        # Compute final stats
        self._transfer_stats.total_time_seconds = (
            time.time() - self._transfer_start_time
        )
        self._transfer_stats.avg_time_per_paragraph = (
            self._transfer_stats.total_time_seconds
            / self._transfer_stats.paragraphs_processed
            if self._transfer_stats.paragraphs_processed > 0
            else 0
        )

        logger.info(
            f"Transfer complete: {self._transfer_stats.paragraphs_processed} paragraphs in "
            f"{self._transfer_stats.total_time_seconds:.1f}s "
            f"(avg {self._transfer_stats.avg_time_per_paragraph:.1f}s/para)"
        )

        # Final cleanup: deduplicate paragraphs and remove incomplete ones
        cleaned_outputs = self._cleanup_document_paragraphs(self._transfer_outputs)

        return "\n\n".join(cleaned_outputs), self._transfer_stats

    def _cleanup_document_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Clean up paragraphs: remove duplicates, incomplete content, etc."""
        from ..utils.nlp import is_sentence_incomplete, get_complete_sentences

        cleaned = []
        seen_starts = {}  # Map of first 50 chars -> full paragraph

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check for duplicate paragraphs (same start)
            para_start = para[:50].lower() if len(para) > 50 else para.lower()
            if para_start in seen_starts:
                existing_text, existing_idx = seen_starts[para_start]
                # Keep the longer/more complete version
                if len(para) > len(existing_text):
                    # Replace with longer version using tracked index
                    cleaned[existing_idx] = para
                    seen_starts[para_start] = (para, existing_idx)
                logger.debug(f"Skipping duplicate paragraph: {para[:50]}...")
                continue

            # Check if paragraph ends incomplete using spaCy
            sentences = split_into_sentences(para)
            para_modified = False
            if sentences:
                last_sent = sentences[-1]
                is_incomplete, reason = is_sentence_incomplete(last_sent)
                if is_incomplete and reason != "no ending punctuation":
                    logger.warning(
                        f"Paragraph ends incomplete ({reason}), truncating: ...{para[-50:]}"
                    )
                    # Keep only complete sentences
                    complete = get_complete_sentences(para)
                    if complete:
                        para = " ".join(complete)
                        para_modified = True
                    else:
                        # Can't salvage - add period if long enough
                        if len(para.split()) > 10:
                            para = para + "."
                            para_modified = True
                        else:
                            continue

            # Check for internal repetition (same sentence repeated)
            if para_modified:
                sentences = split_into_sentences(para)
            if len(sentences) > 1:
                unique_sentences = []
                seen_sents = set()
                for sent in sentences:
                    sent_normalized = sent.strip().lower()
                    if sent_normalized not in seen_sents:
                        seen_sents.add(sent_normalized)
                        unique_sentences.append(sent.strip())
                    else:
                        logger.debug(
                            f"Removing repeated sentence within paragraph: {sent[:40]}..."
                        )
                if len(unique_sentences) < len(sentences):
                    para = " ".join(unique_sentences)

            seen_starts[para_start] = (para, len(cleaned))
            cleaned.append(para)

        return cleaned

    def get_partial_results(self) -> Tuple[str, TransferStats]:
        """Get partial results after an interrupted transfer.

        Returns:
            Tuple of (partial_output, statistics).
        """
        # Compute stats for partial transfer
        if hasattr(self, "_transfer_stats") and hasattr(self, "_transfer_start_time"):
            self._transfer_stats.total_time_seconds = (
                time.time() - self._transfer_start_time
            )
            if self._transfer_stats.paragraphs_processed > 0:
                self._transfer_stats.avg_time_per_paragraph = (
                    self._transfer_stats.total_time_seconds
                    / self._transfer_stats.paragraphs_processed
                )

        outputs = getattr(self, "_transfer_outputs", [])
        stats = getattr(self, "_transfer_stats", TransferStats())

        return "\n\n".join(outputs), stats
