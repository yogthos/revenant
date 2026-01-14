"""Style transfer pipeline using LoRA with RTT neutralization.

This module provides a style transfer pipeline that uses LoRA-adapted models
for consistent style transfer with entailment validation.

Pipeline:
1. RTT neutralization (English → Mandarin HSK5 → Plain English) to strip style
2. Pass neutralized text to LoRA for style application
3. Validate styled output via NLI entailment
4. Apply repetition reduction to fix LLM-speak
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import time

from .lora_generator import AdapterSpec
from .base_generator import GenerationConfig
from .factory import create_style_generator
from ..config import get_adapter_config
from .document_context import DocumentContext, extract_document_context
from ..utils.nlp import (
    split_into_paragraphs,
    split_into_sentences,
    is_heading,
)
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
    from ..persona import get_persona_config, build_persona_prompt
    PERSONA_AVAILABLE = True
except ImportError:
    PERSONA_AVAILABLE = False
    get_persona_config = None

logger = get_logger(__name__)




@dataclass
class TransferConfig:
    """Configuration for style transfer."""

    # Generation settings
    max_tokens: int = 512
    temperature: Optional[float] = None  # None = use lora config, float = CLI override
    top_p: float = 0.9

    # Verification settings
    verify_entailment: bool = True
    entailment_threshold: float = 0.7
    max_hallucinations_before_reject: int = 2  # Trigger repair after this many hallucinations

    # Repair settings
    max_repair_attempts: int = 3
    repair_temperature: float = 0.3  # Lower temperature for repair attempts

    # Post-processing settings
    reduce_repetition: bool = True
    repetition_threshold: int = 3  # Words used 3+ times get replaced

    # Content handling
    pass_headings_unchanged: bool = True  # Don't transform headings
    min_paragraph_words: int = 10  # Skip very short paragraphs

    # Document context settings
    use_document_context: bool = True  # Extract and use document-level context

    # Input format (uses graph-based description matching training format)

    # Length control settings
    max_expansion_ratio: float = 2.5  # Max output/input word ratio before warning
    target_expansion_ratio: float = 1.5  # Target for LoRA generation (1.5 = 50% expansion for author flourish)
    expand_for_texture: bool = False  # Add stronger expansion prompt for texture/flourishes

    # Neutralization settings
    skip_neutralization: bool = False  # If True, skip RTT and use original text as input

    # Perspective settings
    perspective: str = "preserve"  # preserve, first_person_singular, first_person_plural, third_person

    # Structural RAG settings
    use_structural_rag: bool = True  # Enable Structural RAG for rhythm/syntax guidance
    use_structural_grafting: bool = True  # Enable Structural Grafting for argument skeletons
    rag_sample_size: int = 200  # Number of corpus chunks to sample for rhythm pattern analysis

    # Sentence restructuring settings (convert mechanical patterns to organic)
    restructure_sentences: bool = True  # Enable balanced→inverted restructuring

    # Sentence splitting settings (break run-on sentences)
    split_sentences: bool = True  # Enable sentence splitting at conjunction points
    max_sentence_length: int = 50  # Words - split sentences longer than this
    sentence_length_variance: float = 0.3  # Variance factor (0.3 = 70%-130% of max)

    # Grammar correction settings (final post-processing pass)
    correct_grammar: bool = True  # Enable style-safe grammar correction
    grammar_language: str = "en-US"  # Language variant: "en-US" or "en-GB"

    # Persona settings (subjective voice to defeat AI detection)
    use_persona: bool = True  # Enable persona-based prompting
    apply_input_perturbation: bool = True  # Apply 8% noise to match training distribution


@dataclass
class TransferStats:
    """Statistics from a transfer operation."""

    paragraphs_processed: int = 0
    paragraphs_repaired: int = 0
    words_replaced: int = 0
    sentences_restructured: int = 0
    sentences_split: int = 0
    grammar_corrections: int = 0
    total_time_seconds: float = 0.0
    avg_time_per_paragraph: float = 0.0
    entailment_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paragraphs_processed": self.paragraphs_processed,
            "paragraphs_repaired": self.paragraphs_repaired,
            "words_replaced": self.words_replaced,
            "sentences_restructured": self.sentences_restructured,
            "sentences_split": self.sentences_split,
            "grammar_corrections": self.grammar_corrections,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_paragraph": round(self.avg_time_per_paragraph, 2),
            "avg_entailment_score": round(
                sum(self.entailment_scores) / len(self.entailment_scores), 3
            ) if self.entailment_scores else 0.0,
        }


class StyleTransfer:
    """Style transfer using LoRA with RTT neutralization.

    This is the main entry point for style transfer. Pipeline:

    1. RTT neutralize input (English → Mandarin → English)
    2. Pass neutralized text to LoRA for style application
    3. Validate via NLI entailment
    4. Apply repetition reduction

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
        """
        self.config = config or TransferConfig()
        self.author = author_name
        self.verify_fn = verify_fn

        # Convert string provider name to actual LLMProvider object if needed
        if isinstance(critic_provider, str):
            from ..llm.provider import create_critic_provider
            from ..config import load_config
            app_config = load_config()
            self.critic_provider = create_critic_provider(app_config.llm)
        else:
            self.critic_provider = critic_provider

        # Log key config settings
        logger.info(f"StyleTransfer config: expand_for_texture={self.config.expand_for_texture}, "
                    f"target_expansion_ratio={self.config.target_expansion_ratio}")

        # Determine the primary adapter path for config loading
        if adapters:
            primary_adapter_path = adapters[0].path
        else:
            primary_adapter_path = adapter_path

        # Initialize generator using adapter-specific config from config.json
        # This loads temperature, top_p, min_p, repetition_penalty, scale, etc.
        gen_config = GenerationConfig.from_config(primary_adapter_path)
        # Override with CLI-specified temperature if provided
        if self.config.temperature is not None:
            gen_config.temperature = self.config.temperature
        gen_config.skip_cleaning = False  # Always clean output to remove garbage

        # Get backend configuration from adapter config
        adapter_cfg = get_adapter_config(primary_adapter_path)

        # Use factory to create the appropriate generator based on backend
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

        # Initialize repetition reducer for post-processing
        self.repetition_reducer = None
        if self.config.reduce_repetition:
            from ..vocabulary.repetition_reducer import RepetitionReducer
            self.repetition_reducer = RepetitionReducer(
                threshold=self.config.repetition_threshold
            )

        # Initialize sentence restructurer for organic complexity
        self.sentence_restructurer = None
        if self.config.restructure_sentences:
            from ..vocabulary.sentence_restructurer import SentenceRestructurer
            self.sentence_restructurer = SentenceRestructurer()

        # Initialize sentence splitter for run-on sentences
        self.sentence_splitter = None
        if self.config.split_sentences:
            from ..vocabulary.sentence_splitter import SentenceSplitter, SentenceSplitterConfig
            splitter_config = SentenceSplitterConfig(
                max_sentence_length=self.config.max_sentence_length,
                length_variance=self.config.sentence_length_variance,
            )
            self.sentence_splitter = SentenceSplitter(splitter_config)

        # Initialize grammar corrector for final post-processing
        self.grammar_corrector = None
        if self.config.correct_grammar:
            from ..vocabulary.grammar_corrector import GrammarCorrector, GrammarCorrectorConfig
            grammar_config = GrammarCorrectorConfig(language=self.config.grammar_language)
            self.grammar_corrector = GrammarCorrector(grammar_config)

        # Set up entailment verifier if requested
        if self.config.verify_entailment and self.verify_fn is None:
            self.verify_fn = self._create_default_verifier()

        # Initialize RTT neutralizer (local MLX model)
        self._rtt_neutralizer = None

        # Document context (extracted at transfer time)
        self.document_context: Optional[DocumentContext] = None

        # Structural RAG for rhythm/syntax guidance
        self.structural_rag: Optional[StructuralRAG] = None
        if self.config.use_structural_rag:
            if STRUCTURAL_RAG_AVAILABLE:
                self.structural_rag = get_structural_rag(self.author)
                loaded = self.structural_rag.load_patterns(sample_size=self.config.rag_sample_size)
                if loaded > 0:
                    logger.info(f"Structural RAG loaded {loaded} rhythm patterns for {self.author}")
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
                self.structural_grafter = get_structural_grafter(self.author, critic_provider)
                logger.info(f"Structural Grafter initialized for {self.author}")
            else:
                logger.warning("Structural Grafter not available (missing dependencies)")
                self.config.use_structural_grafting = False

    def _rtt_neutralize(self, text: str, max_retries: int = 2) -> Optional[str]:
        """Round-Trip Translation neutralization via Mandarin pivot.

        This matches the training data generation process:
        Step 1 (Scrub): English → Mandarin (HSK3 vocabulary)
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

        return self._rtt_neutralizer.neutralize(text, max_retries=max_retries)

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
            from ..utils.prompts import load_prompt

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
            logger.info(f"TEXTURE EXPANSION result: {input_words} → {output_words} words")

            if response and output_words > input_words:
                expansion = output_words / input_words
                logger.info(f"TEXTURE EXPANSION: expanded by {expansion:.0%}")
                return response.strip()
            else:
                logger.warning(f"Texture expansion returned shorter/equal text ({output_words} vs {input_words}), using original")
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
            from ..utils.prompts import load_prompt

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
                logger.info(f"NARRATIVIZE: {input_words} → {output_words} words (converted to first-person)")
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
            from ..utils.prompts import load_prompt

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

    def _create_default_verifier(self) -> Callable[[str, str], float]:
        """Create default entailment verifier."""
        try:
            import sys
            import os
            import warnings
            import logging
            from io import StringIO

            # Disable tqdm before importing sentence_transformers
            os.environ["TQDM_DISABLE"] = "1"
            from sentence_transformers import CrossEncoder

            # Suppress all output during model loading (position_ids mismatch report)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Suppress transformers logging
                transformers_logger = logging.getLogger("transformers")
                old_level = transformers_logger.level
                transformers_logger.setLevel(logging.ERROR)
                # Suppress stdout/stderr during loading (for LOAD REPORT)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                try:
                    model = CrossEncoder(
                        "cross-encoder/nli-deberta-v3-small",
                        max_length=512,
                    )
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                    transformers_logger.setLevel(old_level)

            def verify(original: str, output: str) -> float:
                """Verify content preservation via entailment."""
                import numpy as np

                # Check if output entails original content
                scores = model.predict([(original, output)], show_progress_bar=False)

                # Model returns raw logits [contradiction, neutral, entailment]
                # Apply softmax to convert to probabilities
                if len(scores.shape) > 1:
                    logits = scores[0]
                else:
                    logits = scores

                # Softmax to get probabilities
                exp_scores = np.exp(logits - np.max(logits))  # subtract max for numerical stability
                probs = exp_scores / np.sum(exp_scores)

                # Return entailment probability (index 2)
                return float(probs[2]) if len(probs) > 2 else float(probs[-1])

            logger.debug("Using NLI-based entailment verification")
            return verify

        except ImportError:
            logger.warning(
                "sentence-transformers not available, using similarity fallback"
            )
            return self._similarity_verifier

    def _similarity_verifier(self, original: str, output: str) -> float:
        """Fallback verifier using keyword overlap."""
        from ..utils.nlp import compute_semantic_similarity

        return compute_semantic_similarity(original, output)

    def transfer_paragraph(
        self,
        paragraph: str,
        previous: Optional[str] = None,
        stats: Optional['TransferStats'] = None,
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
            stats: Optional stats object to update.

        Returns:
            Tuple of (styled_paragraph, entailment_score).
        """
        from ..validation.reference_tracker import extract_references, reinject_references

        # Skip very short paragraphs
        if len(paragraph.split()) < self.config.min_paragraph_words:
            logger.debug(f"Skipping short paragraph: {paragraph[:50]}...")
            return paragraph, 1.0

        word_count = len(paragraph.split())
        logger.debug(f"Translating paragraph: {word_count} words")

        # ========================================
        # STEP 0: Extract and preserve references [^N]
        # ========================================
        # References are stripped before processing and reinjected at the end
        paragraph_clean, ref_map = extract_references(paragraph)

        # Save original for semantic verification (before any expansion)
        original_for_verification = paragraph_clean

        # ========================================
        # STEP 0.5: Texture expansion (optional)
        # ========================================
        # If enabled, use the critic model to add asides, observations, and
        # texture before RTT neutralization. This enriches flat prose.
        if self.config.expand_for_texture:
            logger.info(f"TEXTURE EXPANSION: Starting expansion for {len(paragraph_clean.split())} words")
            paragraph_clean = self._expand_with_texture(paragraph_clean)
            word_count = len(paragraph_clean.split())  # Update word count after expansion
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
            paragraph_clean = self._convert_to_perspective(paragraph_clean, self.config.perspective)
            post_perspective_words = len(paragraph_clean.split())
            logger.info(f"PERSPECTIVE: {pre_perspective_words} → {post_perspective_words} words (→ {self.config.perspective})")

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
            logger.info(f"RTT: Starting neutralization for {len(paragraph_clean.split())} words")
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
                compression_ratio = rtt_output_words / rtt_input_words if rtt_input_words > 0 else 1.0
                logger.info(f"RTT: {rtt_input_words} → {rtt_output_words} words ({compression_ratio:.0%})")

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
                drop_adjectives=True,
            )
            post_perturb_words = len(content_for_generation.split())
            logger.info(f"PERTURBATION: {pre_perturb_words} → {post_perturb_words} words (adjective drops + 8% noise)")

        # ========================================
        # STEP 2: Pass to LoRA for style transformation
        # ========================================
        target_words = int(word_count * self.config.target_expansion_ratio)
        logger.info(f"LORA: content_for_generation={len(content_for_generation.split())} words, target={target_words} words")
        # Token limit needs to be generous to avoid truncation mid-sentence
        # Typically ~1.5 tokens per word, plus some margin for style variation
        # Use 2.5x target words to ensure complete sentences
        max_tokens = max(150, int(target_words * 2.5))

        # Get structural guidance from AUTHOR'S CORPUS (not source text)
        # This is the key to adopting the author's style - their rhythm patterns,
        # sentence lengths, punctuation usage, etc. come from ChromaDB
        structural_guidance = None
        if self.structural_rag:
            guidance = self.structural_rag.get_guidance(paragraph)
            structural_guidance = guidance.format_for_prompt()
            logger.debug(f"Using author structural guidance: {structural_guidance[:100]}...")

        # Get grafting guidance if available
        grafting_guidance = None
        if self.structural_grafter:
            grafting_guidance = self.structural_grafter.get_grafting_guidance(paragraph)
            if grafting_guidance:
                logger.debug(f"Using grafting skeleton: {grafting_guidance.skeleton.format_for_prompt()}")

        # Build persona-injected prompt if enabled
        # CRITICAL: Prompt format must match training format exactly
        final_content = content_for_generation
        use_raw_prompt = False
        if self.config.use_persona and PERSONA_AVAILABLE:
            persona = get_persona_config(self.author)
            final_content = build_persona_prompt(
                content=content_for_generation,
                author=self.author,
                persona=persona,
                vocabulary_palette=persona.adjective_themes[:10],
                structural_guidance=structural_guidance,
                grafting_guidance=grafting_guidance,
                target_words=target_words,  # Pass word count to match training format
                expand_for_texture=self.config.expand_for_texture,
            )
            structural_guidance = None  # Already included in persona prompt
            use_raw_prompt = True  # Use persona prompt directly without additional formatting
            logger.debug(f"Using persona prompt (target={target_words} words)")

        output = self.generator.generate(
            content=final_content,
            author=self.author,
            max_tokens=max_tokens,
            target_words=target_words,
            structural_guidance=structural_guidance,
            raw_prompt=use_raw_prompt,
        )
        lora_output_words = len(output.split())
        lora_input_words = len(content_for_generation.split())
        logger.info(f"LORA OUTPUT: {lora_output_words} words (target was {target_words})")

        # Check if LoRA output matches input (indicates no transformation)
        if output.strip() == paragraph_clean.strip():
            logger.warning("LoRA output identical to original paragraph - no transformation occurred")

        # Check for memorization (output has no semantic overlap with input)
        output_overlap = self._check_content_overlap(content_for_generation, output)
        if output_overlap < 0.1:
            logger.warning(
                f"Possible memorized output detected (only {output_overlap:.0%} content overlap). "
                "Try lowering lora_scale in config.json or using an earlier checkpoint."
            )

        # Track expansion at LoRA stage
        lora_words = len(output.split())
        source_words = len(paragraph.split())
        if lora_words > source_words * self.config.max_expansion_ratio:
            logger.warning(f"LoRA over-expanded: {lora_words} words vs {source_words} source ({lora_words/source_words:.0%})")

        # ========================================
        # STEP 3: Validate styled output (if enabled)
        # ========================================
        logger.info(f"BEFORE VALIDATION: {len(output.split())} words")
        if self.config.verify_entailment:
            from ..validation.semantic_verifier import verify_semantic_preservation
            # Use original (pre-expansion) text for verification
            # We only require the output to preserve the ORIGINAL meaning,
            # not the texture added by the critic
            semantic_result = verify_semantic_preservation(
                source=original_for_verification,
                output=output,
                threshold=self.config.entailment_threshold,
            )

            # Log any issues detected
            issues = semantic_result.get_issues()
            if issues:
                logger.info(f"Semantic issues detected: {', '.join(issues)}")

            # Check for fabricated content (years, citations)
            if semantic_result.fabricated_entities:
                fabricated_str = ', '.join(semantic_result.fabricated_entities[:5])
                logger.info(f"Fabricated content: {fabricated_str}")

            # Trigger repair ONLY for actual hallucinations, not for "missing" entities
            # Missing entities are often just paraphrases (e.g., "roughly" → "about")
            # The repair process generates from original source, which loses styled expansion
            needs_repair = (
                semantic_result.hallucination_count > self.config.max_hallucinations_before_reject
            )

            if needs_repair and semantic_result.missing_entities:
                logger.info(
                    f"Triggering repair: {semantic_result.hallucination_count} hallucinations, "
                    f"{len(semantic_result.missing_entities)} missing entities"
                )
                output = self._repair_missing_entities(
                    source=paragraph,
                    output=output,
                    missing_entities=semantic_result.missing_entities,
                    max_attempts=self.config.max_repair_attempts,
                )
                logger.info(f"AFTER REPAIR: {len(output.split())} words")

        # Ensure output ends with complete sentence
        logger.info(f"BEFORE _ensure_complete_ending: {len(output.split())} words")
        output = self._ensure_complete_ending(output)
        logger.info(f"AFTER _ensure_complete_ending: {len(output.split())} words")

        # ========================================
        # STEP 4: Reinject references [^N]
        # ========================================
        # References were extracted in STEP 0 and are now reattached
        # based on entity matching (e.g., "Einstein" -> "Einstein[^1]")
        if ref_map.has_references():
            output = reinject_references(output, ref_map)
            logger.debug(f"Reinjected {len(ref_map.references)} references")

        # Verify if configured
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
        from ..utils.nlp import get_nlp

        nlp = get_nlp()

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

    def _clean_repair_output(self, text: str) -> str:
        """Clean repair output of meta-commentary and apologies.

        LLMs often prefix repairs with "I apologize" or "Here is the corrected version".
        This strips those out to get just the repaired text.
        """
        import re

        text = text.strip()
        if not text:
            return text

        # Remove common LLM prefixes
        prefixes_to_remove = [
            r'^I apologize[^.]*\.\s*',
            r'^Here is the corrected[^.]*[:.]\s*',
            r'^Here\'s the corrected[^.]*[:.]\s*',
            r'^The corrected text[^.]*[:.]\s*',
            r'^Corrected version[^.]*[:.]\s*',
            r'^Let me (fix|correct)[^.]*\.\s*',
            r'^I\'ve (fixed|corrected)[^.]*\.\s*',
            r'^Sure,?\s*(here[^.]*)?[:.]\s*',
            r'^Of course[^.]*[:.]\s*',
        ]

        for pattern in prefixes_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Also remove trailing meta-commentary
        suffixes_to_remove = [
            r'\s*I hope this[^.]*\.$',
            r'\s*Let me know[^.]*\.$',
            r'\s*Is there anything[^.]*\.$',
        ]

        for pattern in suffixes_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    def _clean_punctuation_artifacts(self, text: str) -> str:
        """Clean up punctuation artifacts from LoRA output and post-processing.

        Fixes common issues like:
        - "—," or ",—" (em-dash combined with comma)
        - ".—" or "—." (em-dash combined with period)
        - Double punctuation
        """
        import re

        # Fix em-dash + punctuation combinations
        text = re.sub(r'—\s*,', ',', text)  # "—," -> ","
        text = re.sub(r',\s*—', ',', text)  # ",—" -> ","
        text = re.sub(r'—\s*\.', '.', text)  # "—." -> "."
        text = re.sub(r'\.\s*—', '.', text)  # ".—" -> "."
        text = re.sub(r'—\s*;', ';', text)  # "—;" -> ";"
        text = re.sub(r';\s*—', ';', text)  # ";—" -> ";"
        text = re.sub(r'—\s*:', ':', text)  # "—:" -> ":"
        text = re.sub(r':\s*—', ':', text)  # ":—" -> ":"

        # Fix double punctuation
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r';\s*;', ';', text)
        text = re.sub(r':\s*:', ':', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # No space before
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Space after

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)

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
        if text[-1] in '.!?':
            return text

        # Find the last complete sentence
        sentences = split_into_sentences(text)
        if not sentences:
            return text

        # Check if last sentence is complete (ends with punctuation)
        complete_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent[-1] in '.!?':
                complete_sentences.append(sent)
            elif sent and len(sent) > 20:
                # Long fragment - try to salvage by adding period
                # Only if it looks like a complete thought
                words = sent.split()
                if len(words) >= 5:
                    complete_sentences.append(sent + '.')
                    logger.warning(f"Added period to incomplete sentence: ...{sent[-30:]}")

        if complete_sentences:
            return ' '.join(complete_sentences)

        # Fallback: add period to entire text
        return text + '.'

    def _repair_missing_entities(
        self,
        source: str,
        output: str,
        missing_entities: List[str],
        max_attempts: int = 3,
    ) -> str:
        """Repair output to include missing entities.

        Uses LoRA generator to re-generate with entity hints while
        maintaining author voice.

        Args:
            source: Original source text.
            output: Styled output to repair.
            missing_entities: List of entity strings to include.
            max_attempts: Max repair attempts.

        Returns:
            Repaired output text.
        """
        if not self.generator or not missing_entities:
            return output

        for attempt in range(max_attempts):
            logger.debug(
                f"Repair attempt {attempt + 1}/{max_attempts}: "
                f"{len(missing_entities)} entities missing"
            )

            try:
                # Create repair content with entity hints
                entity_hint = f"[MUST INCLUDE: {', '.join(missing_entities)}]"
                repair_content = f"{entity_hint}\n\n{source}"

                target_words = len(output.split())
                repaired = self.generator.generate(
                    content=repair_content,
                    author=self.author,
                    target_words=target_words,
                    temperature=self.config.repair_temperature,
                )

                if repaired and len(repaired.split()) > 10:
                    # Check if repair includes missing entities
                    repaired_lower = repaired.lower()
                    entities_found = sum(1 for e in missing_entities if e.lower() in repaired_lower)

                    if entities_found >= len(missing_entities) * 0.5:
                        logger.debug(f"Repair found {entities_found}/{len(missing_entities)} entities")
                        return repaired
                    else:
                        logger.debug(f"Repair only found {entities_found}/{len(missing_entities)} entities")
                else:
                    logger.warning("Repair produced empty/short output")

            except Exception as e:
                logger.warning(f"Repair failed: {e}")
                break

        return output  # Return original if repair failed

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

        # Reset repetition reducer for new document
        if self.repetition_reducer:
            self.repetition_reducer.reset()

        # Split into paragraphs
        paragraphs = split_into_paragraphs(text)

        if not paragraphs:
            logger.warning("No content paragraphs found")
            return text, self._transfer_stats

        # Extract document context for improved generation and critique
        if self.config.use_document_context:
            logger.info("Extracting document context...")
            self.document_context = extract_document_context(
                text,
                llm_provider=self.critic_provider,
            )

        logger.info(f"Transferring {len(paragraphs)} paragraphs")

        previous = None

        for i, para in enumerate(paragraphs):
            if on_progress:
                on_progress(i + 1, len(paragraphs), f"Processing paragraph {i+1}")

            para_start = time.time()

            # Check if paragraph is a heading - pass through unchanged
            para_lines = para.strip().split('\n')
            is_heading_para = self.config.pass_headings_unchanged and len(para_lines) == 1 and is_heading(para_lines[0])

            if is_heading_para:
                logger.debug(f"Passing heading unchanged: {para[:50]}...")
                output = para
                score = 1.0
            else:
                output, score = self.transfer_paragraph(para, previous, self._transfer_stats)

            # Apply repetition reduction only to transformed content, not headings
            if self.repetition_reducer and not is_heading_para:
                output, reduction_stats = self.repetition_reducer.reduce(output)
                self._transfer_stats.words_replaced += reduction_stats.replacements_made

            # Apply sentence restructuring for organic complexity
            if self.sentence_restructurer and not is_heading_para:
                output, restructure_stats = self.sentence_restructurer.restructure(output)
                self._transfer_stats.sentences_restructured += restructure_stats.inversions_applied

            # Apply sentence splitting to break run-on sentences
            if self.sentence_splitter and not is_heading_para:
                output, split_stats = self.sentence_splitter.split(output)
                self._transfer_stats.sentences_split += split_stats.total_splits

            # Apply grammar correction as final step (after sentence splitting)
            if self.grammar_corrector and not is_heading_para:
                output, grammar_stats = self.grammar_corrector.correct(output)
                self._transfer_stats.grammar_corrections += grammar_stats.corrections_applied

            para_time = time.time() - para_start
            logger.debug(f"Paragraph {i+1}: {para_time:.1f}s, score={score:.2f}")

            self._transfer_outputs.append(output)
            previous = output

            self._transfer_stats.paragraphs_processed += 1
            self._transfer_stats.entailment_scores.append(score)

            if score < self.config.entailment_threshold:
                self._transfer_stats.paragraphs_repaired += 1

            # Notify callback with completed paragraph
            if on_paragraph:
                on_paragraph(i, output)

        # Compute final stats
        self._transfer_stats.total_time_seconds = time.time() - self._transfer_start_time
        self._transfer_stats.avg_time_per_paragraph = (
            self._transfer_stats.total_time_seconds / self._transfer_stats.paragraphs_processed
            if self._transfer_stats.paragraphs_processed > 0 else 0
        )

        # Log repetition reduction summary
        if self.repetition_reducer and self._transfer_stats.words_replaced > 0:
            overused = self.repetition_reducer.get_overused_words(limit=5)
            if overused:
                logger.info(
                    f"Repetition reduction: {self._transfer_stats.words_replaced} replacements, "
                    f"top overused: {', '.join(w for w, _ in overused)}"
                )

        # Log sentence restructuring summary
        if self.sentence_restructurer and self._transfer_stats.sentences_restructured > 0:
            logger.info(
                f"Sentence restructuring: {self._transfer_stats.sentences_restructured} inversions applied"
            )

        # Log sentence splitting summary
        if self.sentence_splitter and self._transfer_stats.sentences_split > 0:
            logger.info(
                f"Sentence splitting: {self._transfer_stats.sentences_split} sentences split"
            )

        # Log grammar correction summary
        if self.grammar_corrector and self._transfer_stats.grammar_corrections > 0:
            logger.info(
                f"Grammar correction: {self._transfer_stats.grammar_corrections} corrections applied"
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
                existing = seen_starts[para_start]
                # Keep the longer/more complete version
                if len(para) > len(existing):
                    # Replace with longer version
                    idx = cleaned.index(existing)
                    cleaned[idx] = para
                    seen_starts[para_start] = para
                logger.debug(f"Skipping duplicate paragraph: {para[:50]}...")
                continue

            # Check if paragraph ends incomplete using spaCy
            if para:
                # Get the last sentence and check if it's incomplete
                sentences = split_into_sentences(para)
                if sentences:
                    last_sent = sentences[-1]
                    is_incomplete, reason = is_sentence_incomplete(last_sent)
                    if is_incomplete and reason != "no ending punctuation":
                        logger.warning(f"Paragraph ends incomplete ({reason}), truncating: ...{para[-50:]}")
                        # Keep only complete sentences
                        complete = get_complete_sentences(para)
                        if complete:
                            para = " ".join(complete)
                        else:
                            # Can't salvage - add period if long enough
                            if len(para.split()) > 10:
                                para = para + "."
                            else:
                                continue

            # Check for internal repetition (same sentence repeated)
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
                        logger.debug(f"Removing repeated sentence within paragraph: {sent[:40]}...")
                if len(unique_sentences) < len(sentences):
                    para = " ".join(unique_sentences)

            seen_starts[para_start] = para
            cleaned.append(para)

        return cleaned

    def get_partial_results(self) -> Tuple[str, TransferStats]:
        """Get partial results after an interrupted transfer.

        Returns:
            Tuple of (partial_output, statistics).
        """
        # Compute stats for partial transfer
        if hasattr(self, '_transfer_stats') and hasattr(self, '_transfer_start_time'):
            self._transfer_stats.total_time_seconds = time.time() - self._transfer_start_time
            if self._transfer_stats.paragraphs_processed > 0:
                self._transfer_stats.avg_time_per_paragraph = (
                    self._transfer_stats.total_time_seconds / self._transfer_stats.paragraphs_processed
                )

        outputs = getattr(self, '_transfer_outputs', [])
        stats = getattr(self, '_transfer_stats', TransferStats())

        return "\n\n".join(outputs), stats
