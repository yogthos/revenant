"""Style transfer pipeline using LoRA.

This module provides a style transfer pipeline that uses LoRA-adapted models
for consistent style transfer with a critic/repair loop.

Pipeline:
1. Extract document context (thesis, intent, tone)
2. For each paragraph:
   - Generate styled text (LoRA call)
   - Validate propositions preserved
   - Repair with critic if needed
3. Post-process to reduce repetition
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import time

from .lora_generator import LoRAStyleGenerator, GenerationConfig
from .document_context import DocumentContext, extract_document_context
from ..utils.nlp import (
    split_into_paragraphs,
    split_into_sentences,
    filter_headings,
    is_heading,
)
from ..utils.logging import get_logger
from ..utils.prompts import format_prompt

logger = get_logger(__name__)


@dataclass
class TransferConfig:
    """Configuration for fast style transfer."""

    # Generation settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Verification settings
    verify_entailment: bool = True
    entailment_threshold: float = 0.7

    # Proposition validation settings
    use_proposition_validation: bool = True  # Enable proposition-based validation
    proposition_threshold: float = 0.85  # Min proposition coverage (raised from 0.7 for accuracy)
    anchor_threshold: float = 0.9  # Min content anchor coverage (raised from 0.8 for accuracy)

    # Quality critic settings
    use_quality_critic: bool = True  # Enable quality checking with explicit fix instructions
    word_cluster_threshold: int = 3  # Words used 3+ times trigger warning

    # Repair settings (can be overridden by app config)
    max_repair_attempts: int = 3
    repair_temperature: float = 0.3

    # Post-processing settings
    reduce_repetition: bool = True
    repetition_threshold: int = 3  # Words used 3+ times get replaced

    # Content handling
    pass_headings_unchanged: bool = True  # Don't transform headings
    min_paragraph_words: int = 10  # Skip very short paragraphs

    # Document context settings
    use_document_context: bool = True  # Extract and use document-level context

    # Neutralization settings - convert prose to description before LoRA
    use_neutralization: bool = True  # Neutralize paragraphs before transformation
    neutralization_temperature: float = 0.3  # Low temp for consistent descriptions
    neutralization_min_tokens: int = 300  # Minimum tokens for neutralization output
    neutralization_token_multiplier: float = 1.2  # Multiplier for token calculation

    # Content anchor detection settings
    analogy_min_length: int = 10  # Minimum chars for detected analogies
    detect_phase_transitions: bool = True  # Detect "X transforms into Y" patterns

    # Hallucination detection settings
    hallucination_check_noun_phrases: bool = True  # Check for invented noun phrases
    critical_hallucination_words: str = "death,god,soul,spirit,heaven,hell,divine,eternal"

    # Length control settings
    max_expansion_ratio: float = 1.5  # Max output/input word ratio (1.5 = 50% longer)
    target_expansion_ratio: float = 1.2  # Target for LoRA generation
    truncate_over_expanded: bool = False  # If True, truncate; if False, allow longer output

    # LoRA influence settings
    lora_scale: float = 1.0  # 0.0=base only, 0.5=half, 1.0=full, >1.0=amplified

    # Perspective settings
    perspective: str = "preserve"  # preserve, first_person_singular, first_person_plural, third_person, author_voice_third_person

    # Style polish settings - re-run repaired text through LoRA to restore flow
    use_style_polish: bool = True  # Enable polishing after critic repairs
    polish_temperature: float = 0.4  # Low temp for polish to preserve content


@dataclass
class TransferStats:
    """Statistics from a transfer operation."""

    paragraphs_processed: int = 0
    paragraphs_repaired: int = 0
    quality_issues_found: int = 0
    quality_issues_fixed: int = 0
    words_replaced: int = 0
    total_time_seconds: float = 0.0
    avg_time_per_paragraph: float = 0.0
    entailment_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paragraphs_processed": self.paragraphs_processed,
            "paragraphs_repaired": self.paragraphs_repaired,
            "words_replaced": self.words_replaced,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "avg_time_per_paragraph": round(self.avg_time_per_paragraph, 2),
            "avg_entailment_score": round(
                sum(self.entailment_scores) / len(self.entailment_scores), 3
            ) if self.entailment_scores else 0.0,
        }


class PropositionExtractor:
    """Extract semantic propositions from text.

    Uses lightweight heuristics for speed:
    - Split into sentences
    - Extract key claims
    - Identify entities and relationships
    """

    def extract(self, text: str) -> List[str]:
        """Extract propositions from text.

        Args:
            text: Input text.

        Returns:
            List of proposition strings.
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return [text] if text.strip() else []

        # For now, use sentences as propositions
        # Future: use semantic parsing or LLM extraction
        propositions = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 3:  # Skip very short sentences
                propositions.append(sent)

        return propositions


class StyleTransfer:
    """Style transfer using LoRA with critic/repair loop.

    This is the main entry point for style transfer. Pipeline:

    1. Extract propositions (rule-based or LLM)
    2. Single LoRA generation pass
    3. Lightweight verification
    4. Optional single repair pass

    Example usage:
        transfer = StyleTransfer(
            adapter_path="lora_adapters/sagan",
            author_name="Carl Sagan",
        )

        result = transfer.transfer_document(input_text)
        print(result)
    """

    def __init__(
        self,
        adapter_path: Optional[str],
        author_name: str,
        critic_provider,
        config: Optional[TransferConfig] = None,
        verify_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the fast transfer pipeline.

        Args:
            adapter_path: Path to LoRA adapter directory, or None for base model.
            author_name: Author name for prompts.
            critic_provider: LLM provider for critique/repair (e.g., DeepSeek).
            config: Transfer configuration.
            verify_fn: Optional verification function (original, output) -> score.
        """
        self.config = config or TransferConfig()
        self.author = author_name
        self.verify_fn = verify_fn
        self.critic_provider = critic_provider

        # Initialize generator
        gen_config = GenerationConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            lora_scale=getattr(self.config, 'lora_scale', 1.0),
        )
        self.generator = LoRAStyleGenerator(
            adapter_path=adapter_path,
            config=gen_config,
        )

        # Initialize proposition extractor
        self.prop_extractor = PropositionExtractor()

        # Initialize repetition reducer for post-processing
        self.repetition_reducer = None
        if self.config.reduce_repetition:
            from ..vocabulary.repetition_reducer import RepetitionReducer
            self.repetition_reducer = RepetitionReducer(
                threshold=self.config.repetition_threshold
            )

        # Initialize quality critic for explicit fix instructions
        self.quality_critic = None
        if self.config.use_quality_critic:
            from ..validation.quality_critic import QualityCritic
            self.quality_critic = QualityCritic(
                cluster_threshold=self.config.word_cluster_threshold
            )

        # Initialize proposition validator for semantic fidelity checking
        self.proposition_validator = None
        if self.config.use_proposition_validation:
            from ..validation.proposition_validator import PropositionValidator
            self.proposition_validator = PropositionValidator(
                proposition_threshold=self.config.proposition_threshold,
                anchor_threshold=self.config.anchor_threshold,
                check_noun_phrases=getattr(self.config, 'hallucination_check_noun_phrases', True),
                critical_hallucination_words=getattr(
                    self.config, 'critical_hallucination_words',
                    "death,god,soul,spirit,heaven,hell,divine,eternal"
                ),
            )

        # Set up entailment verifier if requested
        if self.config.verify_entailment and self.verify_fn is None:
            self.verify_fn = self._create_default_verifier()

        # Document context (extracted at transfer time)
        self.document_context: Optional[DocumentContext] = None

        logger.info(f"Using critic provider for repairs: {self.critic_provider.provider_name}")

    def _extract_paragraph_thesis(self, paragraph: str) -> str:
        """Extract the main thesis/point of a paragraph.

        This helps the LoRA understand what it's trying to express overall,
        not just individual propositions.

        Args:
            paragraph: Source paragraph text.

        Returns:
            One-sentence thesis statement.
        """
        if not self.critic_provider:
            # Fallback: use first sentence as thesis
            sentences = split_into_sentences(paragraph)
            return sentences[0] if sentences else ""

        try:
            response = self.critic_provider.call(
                system_prompt="You are a precise summarizer. Extract the ONE main point or thesis of this paragraph in a single sentence. Be specific and concrete. Do not add interpretation.",
                user_prompt=f"Paragraph:\n{paragraph}\n\nMain point (one sentence):",
                temperature=0.1,
                max_tokens=100,
            )
            thesis = response.strip()
            # Clean up common prefixes
            for prefix in ["The main point is that ", "The thesis is that ", "This paragraph argues that "]:
                if thesis.lower().startswith(prefix.lower()):
                    thesis = thesis[len(prefix):]
                    thesis = thesis[0].upper() + thesis[1:] if thesis else thesis
                    break
            logger.debug(f"Extracted paragraph thesis: {thesis[:80]}...")
            return thesis
        except Exception as e:
            logger.warning(f"Thesis extraction failed: {e}")
            sentences = split_into_sentences(paragraph)
            return sentences[0] if sentences else ""

    def _create_default_verifier(self) -> Callable[[str, str], float]:
        """Create default entailment verifier."""
        try:
            import sys
            import warnings
            import logging
            from io import StringIO
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
                # Check if output entails original content
                scores = model.predict([(original, output)])
                # Model returns [contradiction, neutral, entailment]
                if len(scores.shape) > 1:
                    return float(scores[0][2])  # Entailment score
                return float(scores[0])

            logger.info("Using NLI-based entailment verification")
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

    def _neutralize_paragraph(self, paragraph: str) -> str:
        """Convert paragraph to neutral semantic description.

        This matches the training format where the LoRA learned to transform
        descriptions into styled prose.

        Args:
            paragraph: Original paragraph text.

        Returns:
            Neutral description of the paragraph content.
        """
        if not self.critic_provider:
            logger.warning("No critic provider for neutralization, using original text")
            return paragraph

        prompt = format_prompt("neutralize_for_transfer", paragraph=paragraph)

        try:
            # Calculate tokens based on input length - need enough room for full content
            input_words = len(paragraph.split())
            min_tokens = getattr(self.config, 'neutralization_min_tokens', 300)
            token_multiplier = getattr(self.config, 'neutralization_token_multiplier', 1.2)
            neutralize_tokens = max(min_tokens, int(input_words * token_multiplier))

            description = self.critic_provider.call(
                system_prompt="You are a precise content summarizer. Describe what the text says in neutral language. Include ALL names, numbers, and specific details.",
                user_prompt=prompt,
                temperature=self.config.neutralization_temperature,
                max_tokens=neutralize_tokens,
            )

            description = description.strip()
            logger.debug(f"Raw neutralized output: {description[:200]}...")

            # Clean up any meta-language the model might add
            prefixes_to_remove = [
                "The passage ", "This text ", "The text ", "This passage ",
                "Here, ", "In this ", "The author ",
            ]
            for prefix in prefixes_to_remove:
                if description.lower().startswith(prefix.lower()):
                    description = description[len(prefix):]
                    # Capitalize first letter
                    if description:
                        description = description[0].upper() + description[1:]
                    break

            logger.debug(f"Neutralized: {len(paragraph.split())} words -> {len(description.split())} words")
            return description

        except Exception as e:
            logger.warning(f"Neutralization failed: {e}, using original text")
            return paragraph

    def _verify_and_augment_neutralization(
        self,
        neutralized: str,
        source_propositions: List,
    ) -> str:
        """Verify neutralized content contains all propositions and augment if needed.

        This is a critical step to prevent proposition loss during neutralization.
        If key content is missing from the neutralized text, we append it explicitly.

        Args:
            neutralized: Neutralized content description.
            source_propositions: Propositions extracted from source.

        Returns:
            Augmented neutralized content with any missing propositions.
        """
        if not source_propositions:
            return neutralized

        neutralized_lower = neutralized.lower()
        missing_content = []

        for prop in source_propositions:
            # Check if key entities are present
            for entity in prop.entities:
                if entity.lower() not in neutralized_lower:
                    missing_content.append(f"Mention: {entity}")

            # Check if content anchors are present
            for anchor in prop.content_anchors:
                if anchor.must_preserve and anchor.text.lower() not in neutralized_lower:
                    if anchor.anchor_type == "example":
                        missing_content.append(f"Example: {anchor.text}")
                    elif anchor.anchor_type == "statistic":
                        missing_content.append(f"Data: {anchor.text}")
                    elif anchor.anchor_type == "quote":
                        missing_content.append(f"Quote: \"{anchor.text}\"")
                    else:
                        missing_content.append(anchor.text)

            # Check if the core proposition is missing (very low keyword overlap)
            prop_keywords = set(kw.lower() for kw in prop.keywords)
            if prop_keywords:
                neutralized_words = set(neutralized_lower.split())
                overlap = len(prop_keywords & neutralized_words)
                coverage = overlap / len(prop_keywords)
                if coverage < 0.3:  # Less than 30% keyword coverage
                    # Add a summary of this proposition
                    if prop.subject and prop.verb:
                        summary = f"{prop.subject} {prop.verb}"
                        if prop.object:
                            summary += f" {prop.object}"
                        missing_content.append(f"Point: {summary}")

        # Deduplicate and filter
        seen = set()
        unique_missing = []
        for item in missing_content:
            item_lower = item.lower()
            if item_lower not in seen and item_lower not in neutralized_lower:
                seen.add(item_lower)
                unique_missing.append(item)

        if unique_missing:
            logger.warning(f"Neutralization missing {len(unique_missing)} items, augmenting")
            # Append missing content
            augmentation = "\n\nMust include:\n- " + "\n- ".join(unique_missing[:10])  # Limit to 10
            return neutralized + augmentation

        return neutralized

    def _build_constrained_content(
        self,
        original_content: str,
        validation,
        source_propositions: List,
    ) -> str:
        """Build content with explicit constraints from failed validation.

        When regeneration fails due to hallucination, this adds explicit
        instructions about what MUST and MUST NOT be included.

        Args:
            original_content: The neutralized content we tried to generate from.
            validation: ValidationResult showing what went wrong.
            source_propositions: The propositions that must be preserved.

        Returns:
            Content string with explicit constraints prepended.
        """
        constraints = []

        # CRITICAL: What MUST be included (from missing propositions)
        must_include = []
        if validation.missing_propositions:
            for match in validation.missing_propositions[:8]:
                prop = match.proposition
                if prop.text and len(prop.text) < 150:
                    must_include.append(prop.text)
                elif prop.subject and prop.verb:
                    summary = f"{prop.subject} {prop.verb}"
                    if prop.object:
                        summary += f" {prop.object}"
                    must_include.append(summary)

        if must_include:
            constraints.append("MUST EXPRESS THESE IDEAS (do not skip any):")
            for i, item in enumerate(must_include, 1):
                constraints.append(f"  {i}. {item}")

        # Missing entities that must appear
        if validation.missing_entities:
            constraints.append(f"\nMUST INCLUDE these names/terms: {', '.join(validation.missing_entities[:10])}")

        # CRITICAL: What MUST NOT be included (hallucinated content)
        must_not_include = []
        if validation.added_entities:
            must_not_include.extend(validation.added_entities[:5])

        for h in validation.hallucinated_content[:3]:
            if h.content_type == "entity" and h.text not in must_not_include:
                must_not_include.append(h.text)

        if must_not_include:
            constraints.append(f"\nDO NOT MENTION: {', '.join(must_not_include)} (these are not in the source)")

        # If we have source propositions, provide a checklist
        if source_propositions and len(source_propositions) <= 10:
            constraints.append("\nCHECKLIST - Your output must cover ALL of these points:")
            for i, prop in enumerate(source_propositions[:10], 1):
                if hasattr(prop, 'text') and prop.text:
                    short_text = prop.text[:80] + "..." if len(prop.text) > 80 else prop.text
                    constraints.append(f"  [{i}] {short_text}")

        # Add general warning
        constraints.append("\nWARNING: Previous attempt hallucinated content. Stay strictly within the source material.")

        if constraints:
            constraint_block = "\n".join(constraints)
            # Log the constraints for debugging
            logger.info(f"Applying {len(must_include)} must-include, {len(must_not_include)} must-not constraints")
            if must_include:
                logger.debug(f"Must include: {must_include[:3]}...")
            if must_not_include:
                logger.debug(f"Must NOT include: {must_not_include}")
            return f"{constraint_block}\n\n---\nCONTENT TO TRANSFORM:\n{original_content}"
        else:
            return original_content

    def transfer_paragraph(
        self,
        paragraph: str,
        previous: Optional[str] = None,
        stats: Optional['TransferStats'] = None,
    ) -> Tuple[str, float]:
        """Transfer a single paragraph with graph-based validation at each step.

        Architecture:
        1. Build source semantic graph (ground truth)
        2. Neutralize text (if enabled)
        3. Build neutralized graph, compare with source, repair if needed
        4. Pass verified neutralized text to LoRA writer
        5. Build styled graph, compare with source, repair if needed

        Args:
            paragraph: Source paragraph.
            previous: Previous output paragraph for continuity.
            stats: Optional stats object to update.

        Returns:
            Tuple of (styled_paragraph, entailment_score).
        """
        from ..validation.semantic_graph import SemanticGraphBuilder, SemanticGraphComparator

        # Skip very short paragraphs
        if len(paragraph.split()) < self.config.min_paragraph_words:
            logger.debug(f"Skipping short paragraph: {paragraph[:50]}...")
            return paragraph, 1.0

        word_count = len(paragraph.split())
        logger.debug(f"Translating paragraph: {word_count} words")

        # ========================================
        # STEP 1: Build source graph (ground truth)
        # ========================================
        builder = SemanticGraphBuilder(use_rebel=False)
        source_graph = builder.build_from_text(paragraph)
        logger.info(f"Source graph: {len(source_graph.nodes)} propositions, {len(source_graph.edges)} relationships")

        if not source_graph.nodes:
            logger.warning("Could not build source graph, passing through unchanged")
            return paragraph, 1.0

        comparator = SemanticGraphComparator()

        # Extract paragraph thesis - the main point the writer needs to express
        paragraph_thesis = self._extract_paragraph_thesis(paragraph)
        logger.debug(f"Paragraph thesis: {paragraph_thesis[:80]}...")

        # ========================================
        # STEP 2: Generate neutral prose directly from graph
        # ========================================
        # The graph already contains all propositions and relationships.
        # We generate neutral prose from it - this is deterministic and
        # preserves ALL content by construction (no LLM that could lose content).
        content_for_generation = source_graph.to_neutral_prose()
        logger.info(f"Generated neutral prose from graph: {len(content_for_generation.split())} words")

        # Prepend thesis to guide the writer on the main point
        if paragraph_thesis:
            content_for_generation = f"MAIN POINT: {paragraph_thesis}\n\n{content_for_generation}"

        # ========================================
        # STEP 4: Pass verified content to LoRA writer
        # ========================================
        target_words = int(word_count * self.config.target_expansion_ratio)
        max_tokens = max(100, int(target_words * 1.5))  # tokens > words

        # Get context hint for generation (if document context available)
        context_hint = None
        if self.document_context:
            context_hint = self.document_context.to_generation_hint()

        output = self.generator.generate(
            content=content_for_generation,
            author=self.author,
            context=previous,
            max_tokens=max_tokens,
            context_hint=context_hint,
            perspective=getattr(self.config, 'perspective', 'preserve'),
        )

        # Check if LoRA output matches input (indicates no transformation)
        if output.strip() == paragraph.strip():
            logger.warning("LoRA output identical to original paragraph - no transformation occurred")
        elif output.strip() == content_for_generation.strip():
            logger.warning("LoRA output identical to neutralized content - no style applied")

        # Track expansion at LoRA stage
        lora_words = len(output.split())
        source_words = len(paragraph.split())
        if lora_words > source_words * self.config.max_expansion_ratio:
            logger.warning(f"LoRA over-expanded: {lora_words} words vs {source_words} source ({lora_words/source_words:.0%})")

        # ========================================
        # STEP 5: Validate styled output against source graph
        # ========================================
        output, is_valid = self._validate_styled_output(
            source=paragraph,
            output=output,
            source_graph=source_graph,
            builder=builder,
            comparator=comparator,
            previous=previous,
            context_hint=context_hint,
            stats=stats,
            max_attempts=self.config.max_repair_attempts,
        )

        # Track expansion and optionally truncate
        final_words = len(output.split())
        max_allowed_words = int(source_words * self.config.max_expansion_ratio)

        if final_words > max_allowed_words:
            expansion_pct = final_words / source_words
            if self.config.truncate_over_expanded:
                logger.warning(f"Output over-expanded: {final_words} words vs {source_words} source ({expansion_pct:.0%}), truncating")
                # Truncate to allowed length at sentence boundary
                sentences = split_into_sentences(output)
                truncated = []
                current_words = 0
                for sent in sentences:
                    sent_words = len(sent.split())
                    if current_words + sent_words > max_allowed_words:
                        break
                    truncated.append(sent)
                    current_words += sent_words

                if truncated:
                    output = ' '.join(truncated)
                # If no complete sentences fit, keep at least the first sentence
                elif sentences:
                    output = sentences[0]
            else:
                logger.info(f"Output expanded: {final_words} words vs {source_words} source ({expansion_pct:.0%})")

        # Ensure output ends with complete sentence
        output = self._ensure_complete_ending(output)

        # Verify if configured
        score = 1.0
        if self.verify_fn:
            score = self.verify_fn(paragraph, output)

        return output, score

    def _ensure_complete_ending(self, text: str) -> str:
        """Ensure text ends with a complete sentence.

        If text ends mid-sentence, remove the incomplete part.
        """
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

    def _validate_and_repair_neutralization(
        self,
        source: str,
        neutralized: str,
        source_graph,
        builder,
        comparator,
    ) -> str:
        """Validate neutralized text against source graph and repair if needed.

        This ensures the neutralization step doesn't lose propositions before
        we pass content to the LoRA writer.

        Args:
            source: Original source text.
            neutralized: Neutralized text to validate.
            source_graph: Semantic graph of source (ground truth).
            builder: SemanticGraphBuilder instance.
            comparator: SemanticGraphComparator instance.

        Returns:
            Validated (and possibly repaired) neutralized text.
        """
        # Build graph from neutralized text
        neutralized_graph = builder.build_from_text(neutralized)
        diff = comparator.compare(source_graph, neutralized_graph)

        if diff.is_isomorphic:
            logger.info("Neutralization graph matches source - no repair needed")
            return neutralized

        if not diff.has_critical_differences:
            logger.debug("Neutralization has minor differences - acceptable")
            return neutralized

        logger.warning(
            f"Neutralization graph diff: {len(diff.missing_nodes)} missing, "
            f"{len(diff.added_nodes)} added propositions"
        )

        # Repair by appending missing propositions from source
        source_sentences = split_into_sentences(source)
        neutralized_sentences = list(split_into_sentences(neutralized))

        for missing_node in diff.missing_nodes:
            # Find the source sentence containing this proposition
            source_sent = self._find_sentence_for_proposition(missing_node, source_sentences)
            if source_sent:
                # Check if already covered
                already_covered = any(
                    self._sentence_contains_proposition(sent, missing_node)
                    for sent in neutralized_sentences
                )
                if not already_covered:
                    # Append to neutralized content
                    neutralized_sentences.append(source_sent)
                    logger.info(f"Added missing proposition to neutralized: {missing_node.summary()[:40]}")

        repaired = " ".join(neutralized_sentences)

        # Verify repair worked
        repaired_graph = builder.build_from_text(repaired)
        final_diff = comparator.compare(source_graph, repaired_graph)

        if final_diff.is_isomorphic or not final_diff.has_critical_differences:
            logger.info("Neutralization repair successful")
        else:
            logger.warning(f"Neutralization repair incomplete: {len(final_diff.missing_nodes)} still missing")

        return repaired

    def _validate_styled_output(
        self,
        source: str,
        output: str,
        source_graph,
        builder,
        comparator,
        previous: Optional[str] = None,
        context_hint: Optional[str] = None,
        stats: Optional['TransferStats'] = None,
        max_attempts: int = 3,
    ) -> Tuple[str, bool]:
        """Validate styled output against source graph with repair.

        This is the post-LoRA validation step that ensures the styled output
        preserves all propositions from the source.

        Args:
            source: Original source text.
            output: Styled output to validate.
            source_graph: Semantic graph of source (ground truth).
            builder: SemanticGraphBuilder instance.
            comparator: SemanticGraphComparator instance.
            previous: Previous paragraph for context.
            context_hint: Document context hint.
            stats: Optional stats object.
            max_attempts: Maximum repair attempts.

        Returns:
            Tuple of (validated_output, is_valid).
        """
        # Initial comparison
        output_graph = builder.build_from_text(output)
        diff = comparator.compare(source_graph, output_graph)

        if diff.is_isomorphic:
            logger.info("Styled output graph matches source - no repair needed")
            return output, True

        if not diff.has_critical_differences:
            logger.debug("Styled output has minor differences - acceptable")
            return output, True

        logger.info(
            f"Styled output graph diff: {len(diff.missing_nodes)} missing, "
            f"{len(diff.added_nodes)} added, {len(diff.entity_role_errors)} entity errors"
        )

        if not self.critic_provider:
            logger.warning("No critic provider for styled output repair")
            return output, False

        # Use incremental repair
        repaired_output = self._incremental_graph_repair(
            source=source,
            output=output,
            source_graph=source_graph,
            builder=builder,
            comparator=comparator,
            max_attempts_per_error=max_attempts,
        )

        # Final verification
        final_graph = builder.build_from_text(repaired_output)
        final_diff = comparator.compare(source_graph, final_graph)

        # Count errors
        final_error_count = (
            len(final_diff.missing_nodes) + len(final_diff.added_nodes) +
            len(final_diff.entity_role_errors)
        )
        original_error_count = (
            len(diff.missing_nodes) + len(diff.added_nodes) + len(diff.entity_role_errors)
        )

        repair_successful = final_diff.is_isomorphic or not final_diff.has_critical_differences
        if repair_successful:
            logger.info("Styled output repair successful")
        else:
            logger.warning(f"Styled output repair incomplete: {final_error_count} errors remaining")

        # ========================================
        # ALWAYS restyle through LoRA at the end
        # ========================================
        # Pass the clean neutral prose (from graph) to the LoRA, not the mixed
        # repaired output. This gives the model consistent, clean input.
        # The graph contains all propositions that must be expressed.
        try:
            old_temp = self.generator.config.temperature
            self.generator.config.temperature = 0.4

            # Use the source graph's neutral prose - this is cleaner than mixed repaired output
            neutral_prose = source_graph.to_neutral_prose()

            styled_output = self.generator.generate(
                content=neutral_prose,
                author=self.author,
                context=previous,
                max_tokens=max(150, int(len(source.split()) * 1.5)),
                context_hint=context_hint,
                perspective=getattr(self.config, 'perspective', 'preserve'),
            )

            self.generator.config.temperature = old_temp

            # Trust the LoRA - the neutral prose contains all propositions
            if stats:
                stats.quality_issues_fixed += 1
            logger.info("Applied author styling to neutral prose")
            return styled_output, repair_successful

        except Exception as e:
            logger.warning(f"Final restyling failed: {e}")
            return repaired_output, repair_successful

    def _call_critic(
        self,
        source: str,
        current_output: str,
        instructions: List[str],
    ) -> str:
        """Call critic provider to make surgical fixes.

        Args:
            source: Original source text.
            current_output: Current styled output with issues.
            instructions: List of specific fix instructions.

        Returns:
            Repaired text.
        """
        if not instructions:
            return current_output

        instruction_text = "\n".join(f"- {inst}" for inst in instructions)

        # Use context-aware prompt if document context is available
        if self.document_context:
            system_prompt = format_prompt(
                "critic_repair_with_context",
                document_context=self.document_context.to_critic_context(),
                instructions=instruction_text
            )
        else:
            system_prompt = format_prompt(
                "critic_repair_system",
                instructions=instruction_text
            )

        # Don't pass source text to critic - only the styled output
        # This prevents the critic from copying source sentences
        user_prompt = format_prompt(
            "critic_repair_user",
            current_output=current_output
        )

        try:
            # Allow enough tokens for completion (current output + room for fixes)
            # Use 1.5x to ensure sentences can be completed
            current_words = len(current_output.split())
            max_repair_tokens = max(200, int(current_words * 1.5))

            repaired = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.repair_temperature,
                max_tokens=max_repair_tokens,
            )
            logger.debug(f"Critic repair applied: {len(instructions)} fixes")
            return repaired.strip()
        except Exception as e:
            logger.warning(f"Critic repair failed: {e}, keeping original output")
            return current_output

    def _polish_output(
        self,
        source: str,
        repaired_output: str,
        previous: Optional[str] = None,
        context_hint: Optional[str] = None,
        stats: Optional['TransferStats'] = None,
    ) -> str:
        """Re-run repaired output through LoRA to restore author voice.

        After critic repairs, the prose can become choppy and lose the author's
        natural rhythm. This step uses the LoRA to re-express the content while
        preserving all the validated facts and propositions.

        Args:
            source: Original source text (for validation).
            repaired_output: Text after critic repairs.
            previous: Previous paragraph for continuity.
            context_hint: Document context hint.
            stats: Optional stats object.

        Returns:
            Polished text with restored author voice.
        """
        if not self.config.use_style_polish:
            return repaired_output

        # Skip if output is very short
        if len(repaired_output.split()) < 15:
            return repaired_output

        logger.info("Polishing repaired output to restore author voice")

        try:
            # Use the repaired output as the content to transform
            # This preserves the corrected facts while allowing the LoRA
            # to restore natural rhythm and flow
            polish_content = format_prompt(
                "style_polish",
            ) + "\n\n" + repaired_output

            # Save and lower temperature for polish
            old_temp = self.generator.config.temperature
            self.generator.config.temperature = self.config.polish_temperature

            try:
                polished = self.generator.generate(
                    content=polish_content,
                    author=self.author,
                    context=previous,
                    max_tokens=max(150, int(len(repaired_output.split()) * 1.3)),
                    context_hint=context_hint,
                    perspective=getattr(self.config, 'perspective', 'preserve'),
                )
            finally:
                self.generator.config.temperature = old_temp

            # Validate the polish didn't lose content
            # Quick check: ensure key entities are preserved
            source_words = set(w.lower() for w in source.split() if len(w) > 4)
            repaired_words = set(w.lower() for w in repaired_output.split() if len(w) > 4)
            polished_words = set(w.lower() for w in polished.split() if len(w) > 4)

            # Key words from repaired output that were validated
            key_words = repaired_words & source_words
            preserved_in_polish = key_words & polished_words

            # If we lost too many key words, keep the repaired version
            if key_words and len(preserved_in_polish) / len(key_words) < 0.7:
                logger.warning(
                    f"Polish lost content ({len(preserved_in_polish)}/{len(key_words)} key words), "
                    "keeping repaired version"
                )
                return repaired_output

            # Check polish isn't drastically different length
            repaired_len = len(repaired_output.split())
            polished_len = len(polished.split())
            if polished_len < repaired_len * 0.5 or polished_len > repaired_len * 1.8:
                logger.warning(
                    f"Polish changed length too much ({polished_len} vs {repaired_len} words), "
                    "keeping repaired version"
                )
                return repaired_output

            if stats:
                stats.quality_issues_fixed += 1

            logger.debug(f"Polish complete: {repaired_len} -> {polished_len} words")
            return polished.strip()

        except Exception as e:
            logger.warning(f"Polish failed: {e}, keeping repaired version")
            return repaired_output

    def _critic_repair(
        self,
        source: str,
        current_output: str,
        validation,
    ) -> str:
        """Use critic provider to surgically fix content issues.

        Args:
            source: Original source text.
            current_output: Current styled output with issues.
            validation: ValidationResult with specific issues.

        Returns:
            Repaired text with style preserved.
        """
        instructions = []

        # First priority: complete any incomplete sentences
        if validation.has_incomplete_sentences:
            instructions.append("COMPLETE the final sentence - it ends abruptly, add a proper ending")

        # Add missing propositions with FULL text (most important for preservation)
        if validation.missing_propositions:
            for match in validation.missing_propositions[:5]:
                prop = match.proposition
                # Include the full proposition text for precise repair
                if prop.text and len(prop.text) < 200:
                    instructions.append(f"MUST EXPRESS this idea: \"{prop.text}\"")
                elif prop.subject and prop.verb:
                    summary = f"{prop.subject} {prop.verb}"
                    if prop.object:
                        summary += f" {prop.object}"
                    instructions.append(f"MUST INCLUDE: {summary}")

                # Also note specific missing elements
                if match.missing_elements:
                    for elem in match.missing_elements[:2]:
                        instructions.append(f"  - Missing {elem}")

        # Add missing entities with context about what they relate to
        if validation.missing_entities:
            for entity in validation.missing_entities[:5]:
                # Skip if already covered by missing propositions
                if any(entity.lower() in str(inst).lower() for inst in instructions):
                    continue
                # Find the proposition that contains this entity for context
                context = self._find_entity_context(source, entity)
                if context:
                    instructions.append(f"ADD '{entity}' - context: {context}")
                else:
                    instructions.append(f"MENTION '{entity}' naturally in the text")

        if validation.added_entities:
            entities = ", ".join(validation.added_entities[:3])
            instructions.append(f"REMOVE these terms (not relevant): {entities}")

        # Handle hallucinated content - critical issues must be removed
        if validation.hallucinated_content:
            critical_hallucinations = [
                h for h in validation.hallucinated_content if h.severity == "critical"
            ]
            for h in critical_hallucinations[:3]:
                instructions.append(
                    f"REMOVAL REQUIRED: Delete any sentence mentioning '{h.text}' - this was not in the source"
                )

        # Add missing facts with the actual fact content
        if validation.missing_facts:
            for fact in validation.missing_facts[:3]:
                # Skip if already covered
                if any(fact.lower() in str(inst).lower() for inst in instructions):
                    continue
                instructions.append(f"INCLUDE this fact: {fact}")

        if validation.stance_violations:
            for violation in validation.stance_violations[:2]:
                instructions.append(violation)

        return self._call_critic(source, current_output, instructions)

    def _verify_semantic_structure(
        self,
        source: str,
        output: str,
        stats: Optional['TransferStats'] = None,
    ) -> str:
        """Verify semantic structure using graph comparison.

        Builds semantic graphs of source and output, compares them,
        and repairs any structural differences.

        Args:
            source: Original source text.
            output: Generated output text.
            stats: Optional stats object.

        Returns:
            Output text (possibly repaired).
        """
        try:
            from ..validation.semantic_graph import build_and_compare_graphs

            source_graph, output_graph, diff = build_and_compare_graphs(source, output)

            if diff.is_isomorphic:
                logger.debug("Semantic structure check passed: graphs are isomorphic")
                return output

            # Log the differences
            logger.info(f"Semantic structure diff: {len(diff.missing_nodes)} missing, {len(diff.added_nodes)} added")

            if not diff.has_critical_differences:
                logger.debug("No critical structural differences")
                return output

            # Generate repair instructions from the diff
            repair_instructions = diff.to_repair_instructions()

            if repair_instructions and self.critic_provider:
                logger.warning(f"Repairing {len(repair_instructions)} structural issues")

                # Include the graph comparison for context
                graph_context = f"""
SOURCE SEMANTIC STRUCTURE:
{source_graph.to_text_description()}

OUTPUT SEMANTIC STRUCTURE:
{output_graph.to_text_description()}

DIFFERENCES:
{diff.to_text()}
"""
                # Build repair prompt with graph context
                instructions = [
                    "SEMANTIC STRUCTURE REPAIR: The output is missing key propositions or relationships.",
                    "Use the graph comparison below to understand what's missing:",
                    graph_context,
                    "SPECIFIC REPAIRS NEEDED:",
                ] + repair_instructions[:5]

                try:
                    output = self._call_critic(source, output, instructions)
                    if stats:
                        stats.quality_issues_fixed += len(repair_instructions)
                    logger.info("Applied semantic structure repair")
                except Exception as e:
                    logger.warning(f"Semantic structure repair failed: {e}")

        except Exception as e:
            logger.warning(f"Semantic structure verification failed: {e}")

        return output

    def _verify_factual_fidelity(
        self,
        source: str,
        output: str,
        stats: Optional['TransferStats'] = None,
    ) -> Tuple[bool, List[str]]:
        """Use LLM to verify output doesn't make claims not in source.

        This catches subtle semantic drift like inventing etymology,
        adding attributions, or changing the meaning of statements.

        Args:
            source: Original source text.
            output: Generated output text.
            stats: Optional stats object.

        Returns:
            Tuple of (is_faithful, list_of_issues).
        """
        if not self.critic_provider:
            return True, []

        try:
            prompt = f"""Compare these two texts and identify any FACTUAL DIFFERENCES.

SOURCE TEXT:
{source}

OUTPUT TEXT:
{output}

List any facts, claims, or assertions in the OUTPUT that are:
1. NOT stated in the SOURCE (invented information)
2. CONTRADICTING the SOURCE
3. MISATTRIBUTING information

Be very strict. If the output adds ANY new factual claims not in the source, list them.
Focus on: etymology claims, origin claims, definitions, attributions, dates, numbers.

If the output is factually faithful, respond with just: FAITHFUL

Otherwise, list each issue on a new line starting with "ISSUE:"
"""

            response = self.critic_provider.call(
                system_prompt="You are a fact-checker. Compare texts and identify any factual differences or invented claims. Be strict and precise.",
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=500,
            )

            response = response.strip()

            if "FAITHFUL" in response.upper() and "ISSUE:" not in response.upper():
                logger.debug("Factual fidelity check passed")
                return True, []

            # Extract issues
            issues = []
            for line in response.split('\n'):
                line = line.strip()
                if line.upper().startswith("ISSUE:"):
                    issue = line[6:].strip()
                    if issue:
                        issues.append(issue)
                elif line and "not in source" in line.lower():
                    issues.append(line)
                elif line and "invented" in line.lower():
                    issues.append(line)
                elif line and "added" in line.lower():
                    issues.append(line)

            if issues:
                logger.warning(f"Factual fidelity issues: {len(issues)}")
                for issue in issues[:3]:
                    logger.warning(f"  - {issue}")

                if stats:
                    stats.quality_issues_found += len(issues)

            return len(issues) == 0, issues

        except Exception as e:
            logger.warning(f"Factual fidelity check failed: {e}")
            return True, []  # Don't block on errors

    def _incremental_graph_repair(
        self,
        source: str,
        output: str,
        source_graph,
        builder,
        comparator,
        max_attempts_per_error: int = 2,
    ) -> str:
        """Repair output with deterministic, targeted fixes.

        This is a single-pass repair algorithm:
        1. Identify all errors from graph diff
        2. For each error, make ONE targeted fix
        3. Stop when all errors processed (no retries on same error)

        The algorithm is deterministic because:
        - Each error is processed exactly once
        - Fixes are direct text operations where possible
        - LLM is only used for specific rewrites, not open-ended generation
        - No loops or retries that could cause infinite iteration

        Args:
            source: Original source text.
            output: Current output text.
            source_graph: Ground truth semantic graph.
            builder: SemanticGraphBuilder instance.
            comparator: SemanticGraphComparator instance.
            max_attempts_per_error: Max LLM attempts for complex fixes.

        Returns:
            Repaired output text.
        """
        source_sentences = split_into_sentences(source)
        output_sentences = list(split_into_sentences(output))

        # Get the diff - this tells us exactly what needs to be fixed
        output_graph = builder.build_from_text(output)
        diff = comparator.compare(source_graph, output_graph)

        if diff.is_isomorphic:
            return output  # Nothing to fix

        errors = self._prioritize_errors(diff)
        logger.info(f"Repairing {len(errors)} graph errors")

        # Track what we've fixed to avoid duplicate work
        fixed_propositions = set()
        removed_sentences = set()
        # Track inserted sentences so cleanup doesn't remove them
        inserted_sentences = set()

        for error in errors:
            error_type = error["type"]
            error_data = error["data"]

            if error_type == "missing":
                # Missing proposition: find source sentence and add it
                prop_id = f"{error_data.subject}|{error_data.predicate}"
                if prop_id in fixed_propositions:
                    continue

                # Find the source sentence containing this proposition
                source_sent = self._find_sentence_for_proposition(
                    error_data, source_sentences
                )

                if not source_sent:
                    logger.warning(f"Could not find source sentence for: {error_data.summary()[:50]}")
                    continue

                # Check if proposition is already covered by an output sentence
                # (not just exact string match - check semantic coverage)
                already_covered = False
                for out_sent in output_sentences:
                    if self._sentence_contains_proposition(out_sent, error_data):
                        already_covered = True
                        logger.debug(f"Proposition already covered: {error_data.summary()[:40]}")
                        break

                if already_covered:
                    fixed_propositions.add(prop_id)
                    continue

                # Insert at position determined by graph structure
                insert_pos = self._find_insertion_position(
                    error_data, output_sentences, source_graph, output_graph
                )
                output_sentences.insert(insert_pos, source_sent)
                inserted_sentences.add(source_sent.lower().strip())
                fixed_propositions.add(prop_id)
                logger.info(f"Inserted missing proposition at pos {insert_pos}: {error_data.summary()[:40]}")

            elif error_type == "added":
                # Hallucinated content: remove the sentence containing it
                prop_text = error_data.text
                for i, sent in enumerate(output_sentences):
                    if i in removed_sentences:
                        continue
                    # Check if this sentence contains the hallucination
                    if self._sentence_contains_proposition(sent, error_data):
                        removed_sentences.add(i)
                        logger.debug(f"Marked for removal: {sent[:40]}...")
                        break

            elif error_type == "entity":
                # Entity error: targeted rewrite of specific sentence
                entity = error_data.entity
                source_role = error_data.source_role

                # Find sentence mentioning this entity
                for i, sent in enumerate(output_sentences):
                    if i in removed_sentences:
                        continue
                    if entity.lower() in sent.lower():
                        # Try direct fix first (for role_loss, add the role)
                        if error_data.error_type == "role_loss":
                            # Find source sentence with correct role
                            source_sent = self._find_sentence_with_entity_role(
                                entity, source_role, source_sentences
                            )
                            if source_sent:
                                output_sentences[i] = source_sent
                                logger.debug(f"Fixed entity role: {entity}")
                        elif error_data.error_type == "conflation":
                            # Remove the conflating sentence
                            removed_sentences.add(i)
                            logger.debug(f"Removed conflation: {entity}")
                        break

        # Remove marked sentences (in reverse order to preserve indices)
        for i in sorted(removed_sentences, reverse=True):
            if i < len(output_sentences):
                output_sentences.pop(i)

        # Log repair summary
        logger.info(f"Repair summary: {len(inserted_sentences)} sentences inserted, {len(removed_sentences)} removed")

        # Cleanup and return - pass inserted_sentences so we don't accidentally remove them
        final_sentences = self._cleanup_repaired_sentences(output_sentences, inserted_sentences)
        return " ".join(final_sentences)

    def _find_sentence_for_proposition(
        self, prop_node, sentences: List[str]
    ) -> Optional[str]:
        """Find the source sentence that contains a proposition."""
        # The proposition node has the original sentence text
        if prop_node.text:
            for sent in sentences:
                if sent.strip() == prop_node.text.strip():
                    return sent
                # Also check for significant overlap
                if self._sentence_overlap(sent, prop_node.text) > 0.7:
                    return sent

        # Fallback: find by subject/predicate
        for sent in sentences:
            sent_lower = sent.lower()
            if (prop_node.subject and prop_node.subject.lower() in sent_lower and
                prop_node.predicate and prop_node.predicate.lower() in sent_lower):
                return sent

        return None

    def _find_insertion_position(
        self,
        prop_node,
        output_sentences: List[str],
        source_graph,
        output_graph,
    ) -> int:
        """Find insertion position using graph structure to maintain narrative order.

        Uses the SOURCE graph node IDs (P1, P2, P3...) which reflect original sentence order.
        Finds where to insert a missing node by looking at which source nodes are already
        in the output and inserting at the correct relative position.
        """
        # Get the node ID (e.g., "P2")
        missing_id = prop_node.id

        try:
            missing_num = int(missing_id[1:])  # "P2" -> 2
        except (ValueError, IndexError):
            return len(output_sentences)

        # Find predecessor in source graph (what points TO this node)
        predecessor_id = None
        for edge in source_graph.edges:
            if edge.target_id == missing_id:
                predecessor_id = edge.source_id
                break

        # Find successor in source graph (what this node points TO)
        successor_id = None
        for edge in source_graph.edges:
            if edge.source_id == missing_id:
                successor_id = edge.target_id
                break

        # Strategy 1: Insert after predecessor if it's in output
        if predecessor_id:
            pred_node = source_graph.get_node(predecessor_id)
            if pred_node:
                for i, sent in enumerate(output_sentences):
                    if self._sentence_contains_proposition(sent, pred_node):
                        return i + 1  # Insert after predecessor

        # Strategy 2: Insert before successor if it's in output
        if successor_id:
            succ_node = source_graph.get_node(successor_id)
            if succ_node:
                for i, sent in enumerate(output_sentences):
                    if self._sentence_contains_proposition(sent, succ_node):
                        return i  # Insert before successor

        # Strategy 3: Find any SOURCE node in output with higher ID and insert before it
        for source_node in sorted(source_graph.nodes, key=lambda n: int(n.id[1:]) if n.id[1:].isdigit() else 999):
            try:
                source_num = int(source_node.id[1:])
                if source_num > missing_num:
                    # Check if this source node's sentence is in output
                    for i, sent in enumerate(output_sentences):
                        if self._sentence_contains_proposition(sent, source_node):
                            return i  # Insert before this higher-numbered source node
            except (ValueError, IndexError):
                pass

        # Strategy 4: If missing node has lower ID than any source node in output, insert at 0
        for source_node in source_graph.nodes:
            try:
                source_num = int(source_node.id[1:])
                if source_num > missing_num:
                    for sent in output_sentences:
                        if self._sentence_contains_proposition(sent, source_node):
                            return 0  # There's a higher node in output, so insert at beginning
            except (ValueError, IndexError):
                pass

        # Default: append at end
        return len(output_sentences)

    def _sentence_contains_proposition(self, sentence: str, prop_node) -> bool:
        """Check if a sentence contains a specific proposition."""
        sent_lower = sentence.lower()

        # Check for subject and predicate
        has_subject = prop_node.subject and prop_node.subject.lower() in sent_lower
        has_predicate = prop_node.predicate and prop_node.predicate.lower() in sent_lower

        if has_subject and has_predicate:
            return True

        # Check for significant keyword overlap
        if prop_node.text:
            return self._sentence_overlap(sentence, prop_node.text) > 0.6

        return False

    def _find_sentence_with_entity_role(
        self, entity: str, role: str, sentences: List[str]
    ) -> Optional[str]:
        """Find a source sentence where an entity performs a specific role."""
        entity_lower = entity.lower()
        role_words = set(role.lower().split())

        for sent in sentences:
            sent_lower = sent.lower()
            if entity_lower in sent_lower:
                # Check if role words are present
                sent_words = set(sent_lower.split())
                if role_words & sent_words:  # Any overlap
                    return sent

        return None

    def _cleanup_repaired_sentences(
        self,
        sentences: List[str],
        protected_sentences: Optional[set] = None,
    ) -> List[str]:
        """Clean up sentences after repair: dedupe, remove incomplete, etc.

        Args:
            sentences: List of sentences to clean up.
            protected_sentences: Set of sentence texts (lowercase, stripped) that were
                inserted during repair and should NOT be filtered out.

        Returns:
            Cleaned list of sentences.
        """
        from ..utils.nlp import is_sentence_incomplete

        protected = protected_sentences or set()
        cleaned = []
        seen = set()

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            sent_normalized = sent.lower().strip()
            is_protected = sent_normalized in protected

            # Skip duplicate sentences (even protected ones shouldn't appear twice)
            if sent_normalized in seen:
                logger.debug(f"Removing duplicate sentence: {sent[:50]}...")
                continue

            # Check for incomplete sentences using spaCy POS analysis
            # BUT skip this check for protected sentences (source sentences are complete)
            if not is_protected:
                is_incomplete, reason = is_sentence_incomplete(sent)
                if is_incomplete:
                    # Try to salvage by adding punctuation if it looks nearly complete
                    if reason == "no ending punctuation" and len(sent.split()) > 8:
                        sent = sent + "."
                    else:
                        logger.warning(f"Removing incomplete sentence ({reason}): {sent[:50]}...")
                        continue

            # Skip sentences that are mostly duplicated content from another
            # BUT for protected sentences, we need to be smarter - they contain
            # critical propositions that may not be fully covered by styled sentences
            is_subset = False
            for existing in cleaned:
                # Check if this sentence is largely contained in another
                if len(sent) > 20 and sent[:20].lower() in existing.lower():
                    overlap = self._sentence_overlap(sent, existing)
                    if overlap > 0.8:
                        if is_protected:
                            # Protected sentence overlaps with existing styled sentence
                            # Keep the protected one (source) since it has the full proposition
                            # and REPLACE the styled one
                            idx = cleaned.index(existing)
                            cleaned[idx] = sent
                            logger.info(f"Replaced overlapping styled sentence with source: {sent[:50]}...")
                            is_subset = True  # Don't add again
                        else:
                            logger.debug(f"Removing overlapping sentence: {sent[:50]}...")
                            is_subset = True
                        break

            if is_subset:
                # For protected sentences, we already replaced in the loop above
                # For non-protected, we skip
                if is_protected:
                    seen.add(sent_normalized)
                continue

            seen.add(sent_normalized)
            cleaned.append(sent)

        return cleaned

    def _sentence_overlap(self, sent1: str, sent2: str) -> float:
        """Calculate word overlap between two sentences."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        return intersection / min(len(words1), len(words2))

    def _prioritize_errors(self, diff) -> List[dict]:
        """Prioritize errors for repair.

        Order:
        1. Entity role errors (most critical - change meaning)
        2. Missing propositions (content loss)
        3. Added propositions (hallucinations)
        """
        errors = []

        # Entity errors first - these are the most critical
        for i, err in enumerate(diff.entity_role_errors):
            errors.append({
                "type": "entity",
                "priority": 0,
                "data": err,
                "description": f"{err.error_type}: {err.entity} - {err.source_role}",
            })

        # Missing propositions
        for i, node in enumerate(diff.missing_nodes):
            errors.append({
                "type": "missing",
                "priority": 1,
                "data": node,
                "description": f"Missing: {node.summary()}",
            })

        # Added propositions (hallucinations)
        for i, node in enumerate(diff.added_nodes):
            errors.append({
                "type": "added",
                "priority": 2,
                "data": node,
                "description": f"Remove: {node.summary()}",
            })

        # Sort by priority
        errors.sort(key=lambda x: x["priority"])
        return errors

    def _verify_semantic_coherence(
        self,
        source: str,
        output: str,
        thesis: str,
        stats: Optional['TransferStats'] = None,
    ) -> str:
        """Final check that overall meaning and thesis are preserved.

        This is a holistic check after proposition-level validation to ensure
        the output conveys the same overall message as the source.

        Args:
            source: Original source text.
            output: Current styled output.
            thesis: Extracted thesis statement.
            stats: Optional stats object to update.

        Returns:
            Output text (possibly repaired if thesis was missing).
        """
        # Extract key concepts from thesis
        from ..utils.nlp import get_nlp
        nlp = get_nlp()

        thesis_doc = nlp(thesis)
        output_lower = output.lower()

        # Get key content words from thesis
        thesis_keywords = set()
        for token in thesis_doc:
            if token.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not token.is_stop:
                thesis_keywords.add(token.lemma_.lower())

        # Check keyword coverage
        found_keywords = sum(1 for kw in thesis_keywords if kw in output_lower)
        coverage = found_keywords / len(thesis_keywords) if thesis_keywords else 1.0

        # Also check named entities from thesis
        thesis_entities = [ent.text.lower() for ent in thesis_doc.ents]
        entities_found = sum(1 for ent in thesis_entities if ent in output_lower)
        entity_coverage = entities_found / len(thesis_entities) if thesis_entities else 1.0

        # Combined coherence score
        coherence_score = (coverage * 0.6 + entity_coverage * 0.4)

        if coherence_score >= 0.7:
            logger.debug(f"Semantic coherence check passed: {coherence_score:.0%}")
            return output

        # Thesis is not well-preserved - need repair
        logger.warning(
            f"Semantic coherence low ({coherence_score:.0%}): "
            f"thesis keywords {found_keywords}/{len(thesis_keywords)}, "
            f"entities {entities_found}/{len(thesis_entities)}"
        )

        if stats:
            stats.quality_issues_found += 1

        # Repair to restore thesis
        if self.critic_provider:
            try:
                instructions = [
                    f"CRITICAL: The output must clearly express this main point: \"{thesis}\"",
                    "Ensure the core argument/thesis is evident, not just supporting details",
                ]

                # Add specific missing keywords if any
                missing_keywords = [kw for kw in thesis_keywords if kw not in output_lower]
                if missing_keywords:
                    instructions.append(f"Consider including key concepts: {', '.join(missing_keywords[:5])}")

                repaired = self._call_critic(source, output, instructions)

                if stats:
                    stats.quality_issues_fixed += 1

                logger.info("Applied semantic coherence repair")
                return repaired

            except Exception as e:
                logger.warning(f"Semantic coherence repair failed: {e}")

        return output

    def _find_entity_context(self, source: str, entity: str) -> Optional[str]:
        """Find the sentence containing an entity to provide context."""
        sentences = split_into_sentences(source)
        for sent in sentences:
            if entity.lower() in sent.lower():
                # Return a shortened version of the sentence as context
                if len(sent) > 100:
                    # Find the clause containing the entity
                    words = sent.split()
                    entity_words = entity.lower().split()
                    for i, word in enumerate(words):
                        if word.lower() in entity_words:
                            start = max(0, i - 5)
                            end = min(len(words), i + len(entity_words) + 5)
                            return "..." + " ".join(words[start:end]) + "..."
                return sent[:100] + "..." if len(sent) > 100 else sent
        return None

    def _quality_critic_repair(
        self,
        source: str,
        current_output: str,
        critique,
    ) -> str:
        """Use critic provider to fix quality issues.

        Args:
            source: Original source text.
            current_output: Current styled output with issues.
            critique: QualityCritique with specific issues.

        Returns:
            Repaired text with style preserved.
        """
        instructions = []
        for issue in critique.issues:
            if issue.fix_instruction:
                instructions.append(issue.fix_instruction)

        # Always check for grammar/completeness
        instructions.append("FIX any incomplete or ungrammatical sentences")

        return self._call_critic(source, current_output, instructions)

    def _repair(
        self,
        original: str,
        current: str,
        propositions: List[str],
        previous: Optional[str],
    ) -> Tuple[str, float]:
        """Single repair pass for meaning preservation.

        Args:
            original: Original paragraph.
            current: Current (failed) output.
            propositions: Extracted propositions.
            previous: Previous paragraph.

        Returns:
            Tuple of (repaired_output, new_score).
        """
        logger.info("Attempting repair...")

        # Generate with lower temperature and stricter prompt
        old_temp = self.generator.config.temperature
        self.generator.config.temperature = self.config.repair_temperature

        # Use stricter system prompt for repair
        repair_system = format_prompt("repair_strict", author=self.author)

        try:
            # Estimate tokens based on input
            input_words = len(original.split())
            max_tokens = max(100, int(input_words * 2.0))

            # Get context hint for generation (if document context available)
            context_hint = None
            if self.document_context:
                context_hint = self.document_context.to_generation_hint()

            output = self.generator.generate(
                content=original,
                author=self.author,
                context=previous,
                system_override=repair_system,
                max_tokens=max_tokens,
                context_hint=context_hint,
                perspective=getattr(self.config, 'perspective', 'preserve'),
            )
        finally:
            self.generator.config.temperature = old_temp

        # Re-verify
        score = self.verify_fn(original, output) if self.verify_fn else 1.0
        logger.info(f"Repair result: score={score:.2f}")

        return output, score

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
            if self.config.pass_headings_unchanged and len(para_lines) == 1 and is_heading(para_lines[0]):
                logger.debug(f"Passing heading unchanged: {para[:50]}...")
                output = para
                score = 1.0
            else:
                output, score = self.transfer_paragraph(para, previous, self._transfer_stats)

            # Apply repetition reduction (only to transformed content, not headings)
            if self.repetition_reducer and score < 1.0:
                output, reduction_stats = self.repetition_reducer.reduce(output)
                self._transfer_stats.words_replaced += reduction_stats.replacements_made

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

    def switch_author(self, adapter_path: str, author_name: str) -> None:
        """Switch to a different author.

        Args:
            adapter_path: Path to new adapter.
            author_name: New author name.
        """
        self.generator.switch_adapter(adapter_path)
        self.author = author_name
        logger.info(f"Switched to author: {author_name}")


def create_style_transfer(
    adapter_path: str,
    author_name: str,
    verify: bool = True,
    temperature: float = 0.7,
) -> StyleTransfer:
    """Convenience function to create a style transfer pipeline.

    Args:
        adapter_path: Path to LoRA adapter.
        author_name: Author name.
        verify: Whether to enable entailment verification.
        temperature: Generation temperature.

    Returns:
        Configured StyleTransfer instance.
    """
    config = TransferConfig(
        verify_entailment=verify,
        temperature=temperature,
    )

    return StyleTransfer(
        adapter_path=adapter_path,
        author_name=author_name,
        config=config,
    )
