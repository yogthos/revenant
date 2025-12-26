"""Template-based style transfer pipeline with semantic preservation.

Orchestrates the full style transfer process:
1. Extract transferable style (vocabulary, transitions, patterns) from corpus
2. Extract propositions from input (semantic content to preserve)
3. Build context-aware prompts combining templates + style + discourse
4. Generate with LLM using structural scaffolding
5. Verify style match using Burrows' Delta
6. Repair if needed
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Callable
import random

from ..utils.nlp import get_nlp, split_into_sentences, calculate_burstiness
from ..utils.logging import get_logger
from .models import (
    SentenceTemplate,
    CorpusStatistics,
    VocabularyProfile,
    SentenceType,
    RhetoricalRole,
    LogicalRelation,
)
from .storage import TemplateStore
from .extractor import SkeletonExtractor
from .filler import SlotFiller, Proposition, FilledSentence, TemplateMatcher
from .vocabulary import TechnicalTermExtractor, GeneralWordMapper
from .validator import (
    SentenceValidator,
    SentenceRepairer,
    ParagraphValidator,
    ParagraphRepairer,
)
from .statistics import SentenceClassifier

# New imports for semantic style transfer
from .style_vocabulary import VocabularyClassifier, TemplateAbstractor, ClassifiedVocabulary
from .fingerprint import StyleFingerprintExtractor, StyleFingerprint, StyleVerifier
from .discourse import (
    DiscourseAnalyzer,
    DocumentPosition,
    ParagraphRole,
    SentenceRole,
    DiscourseRelation,
    SentenceContext,
    ParagraphContext,
    DISCOURSE_MARKERS,
)

logger = get_logger(__name__)


@dataclass
class TransferConfig:
    """Configuration for style transfer."""
    author: str
    max_sentence_repair_iterations: int = 3
    max_paragraph_repair_iterations: int = 2
    min_template_match_score: float = 0.3
    enforce_rhythm: bool = True
    enforce_burstiness: bool = True
    preserve_paragraph_structure: bool = True
    n_template_candidates: int = 5

    # Semantic style transfer options
    use_semantic_transfer: bool = True  # Use new semantic-preserving approach
    style_strictness: float = 0.6       # How strictly to enforce style (0-1)
    delta_threshold: float = 2.0        # Max Burrows' Delta for acceptance
    use_llm_generation: bool = True     # Use LLM for generation (vs template filling)
    llm_generator: Optional[Callable[[str], str]] = None  # LLM generation function


@dataclass
class TransferResult:
    """Result of a paragraph transfer."""
    original: str
    transferred: str
    sentences: List[FilledSentence]
    technical_terms_preserved: Set[str]
    validation_passed: bool
    burstiness: float
    corpus_similarity: float


class TemplateBasedTransfer:
    """Main pipeline for template-based style transfer."""

    def __init__(
        self,
        store: TemplateStore,
        config: TransferConfig
    ):
        """Initialize transfer pipeline.

        Args:
            store: Template store with indexed corpus.
            config: Transfer configuration.
        """
        self._nlp = None
        self.store = store
        self.config = config

        # Load author resources
        self.corpus_stats = store.get_statistics(config.author)
        self.vocabulary = store.get_vocabulary(config.author)

        if not self.corpus_stats:
            logger.warning(f"No corpus statistics found for {config.author}")
            self.corpus_stats = CorpusStatistics()

        if not self.vocabulary:
            logger.warning(f"No vocabulary profile found for {config.author}")
            self.vocabulary = VocabularyProfile()

        # Initialize components
        self.extractor = SkeletonExtractor()
        self.classifier = SentenceClassifier()
        self.term_extractor = TechnicalTermExtractor()
        self.filler = SlotFiller(self.vocabulary)
        self.matcher = TemplateMatcher()

        self.sentence_validator = SentenceValidator(self.corpus_stats, self.vocabulary)
        self.sentence_repairer = SentenceRepairer(self.corpus_stats, self.vocabulary)
        self.paragraph_validator = ParagraphValidator(self.corpus_stats, self.vocabulary)
        self.paragraph_repairer = ParagraphRepairer(self.corpus_stats, self.vocabulary)

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def transfer_paragraph(
        self,
        paragraph: str,
        paragraph_position: str = "BODY"
    ) -> TransferResult:
        """Transfer a single paragraph.

        Args:
            paragraph: Input paragraph.
            paragraph_position: Position in document (INTRO/BODY/CONCLUSION).

        Returns:
            TransferResult with transferred text.
        """
        logger.debug(f"=== Starting paragraph transfer ({paragraph_position}) ===")
        logger.debug(f"Input: '{paragraph[:80]}...'")

        # Extract technical terms to preserve
        technical_terms = self.term_extractor.extract(paragraph)
        logger.info(f"Preserving {len(technical_terms)} technical terms")
        logger.debug(f"Technical terms: {list(technical_terms)[:10]}")

        # Split into sentences
        source_sentences = split_into_sentences(paragraph)
        logger.debug(f"Split into {len(source_sentences)} sentences")

        if not source_sentences:
            return TransferResult(
                original=paragraph,
                transferred=paragraph,
                sentences=[],
                technical_terms_preserved=technical_terms,
                validation_passed=False,
                burstiness=0.0,
                corpus_similarity=0.0,
            )

        # Plan sentence sequence
        sentence_plan = self._plan_sentence_sequence(source_sentences, paragraph_position)

        # Transfer each sentence
        transferred_sentences = []
        used_words: Set[str] = set()

        for i, (source, plan) in enumerate(zip(source_sentences, sentence_plan)):
            filled = self._transfer_sentence(
                source=source,
                sentence_type=plan["type"],
                rhetorical_role=plan["role"],
                logical_relation=plan["relation"],
                technical_terms=technical_terms,
                used_words=used_words,
                position=i,
                total=len(source_sentences),
            )

            if filled:
                transferred_sentences.append(filled)

                # Track used words
                for word in filled.text.split():
                    used_words.add(word.lower())

        # Combine sentences
        transferred_text = " ".join([s.text for s in transferred_sentences])

        # Validate and repair paragraph
        validation = self.paragraph_validator.validate(
            transferred_text, technical_terms
        )

        if not validation.is_valid:
            logger.info(f"Paragraph validation failed: {[i.type for i in validation.issues]}")
            transferred_text, repair_success = self.paragraph_repairer.repair(
                transferred_text,
                validation,
                technical_terms,
                self.config.max_paragraph_repair_iterations,
            )
            logger.info(f"Paragraph repair {'succeeded' if repair_success else 'failed'}")

        # Apply rhythm enforcement if needed
        if self.config.enforce_rhythm:
            transferred_text = self._enforce_rhythm(transferred_text)

        # Final validation
        final_validation = self.paragraph_validator.validate(
            transferred_text, technical_terms
        )

        # Compute burstiness
        final_sentences = split_into_sentences(transferred_text)
        burstiness = calculate_burstiness(final_sentences) if len(final_sentences) > 1 else 0

        return TransferResult(
            original=paragraph,
            transferred=transferred_text,
            sentences=transferred_sentences,
            technical_terms_preserved=technical_terms,
            validation_passed=final_validation.is_valid,
            burstiness=burstiness,
            corpus_similarity=final_validation.corpus_similarity,
        )

    def transfer_document(
        self,
        paragraphs: List[str],
        paragraph_callback=None
    ) -> List[TransferResult]:
        """Transfer an entire document.

        Args:
            paragraphs: List of input paragraphs.
            paragraph_callback: Optional callback after each paragraph.

        Returns:
            List of TransferResults.
        """
        results = []

        for i, para in enumerate(paragraphs):
            # Determine paragraph position
            if i == 0:
                position = "INTRO"
            elif i == len(paragraphs) - 1:
                position = "CONCLUSION"
            else:
                position = "BODY"

            logger.info(f"Transferring paragraph {i+1}/{len(paragraphs)} ({position})")

            result = self.transfer_paragraph(para, position)
            results.append(result)

            if paragraph_callback:
                paragraph_callback(i, result)

        return results

    def _plan_sentence_sequence(
        self,
        sentences: List[str],
        paragraph_position: str
    ) -> List[Dict]:
        """Plan the sequence of sentence types and roles.

        Uses transition probabilities to create natural flow.

        Args:
            sentences: Source sentences.
            paragraph_position: Paragraph position in document.

        Returns:
            List of plans with type, role, and relation.
        """
        plans = []
        previous_role = None

        for i, sent in enumerate(sentences):
            # Determine sentence type based on position
            if i == 0:
                sent_type = SentenceType.OPENER
            elif i == len(sentences) - 1:
                sent_type = SentenceType.CLOSER
            else:
                sent_type = SentenceType.BODY

            # Classify source sentence role
            source_role = self.classifier.classify_role(sent, i, len(sentences))
            source_relation = self.classifier.classify_relation(sent)

            # Get likely roles from transition matrix
            if previous_role:
                likely_roles = self.store.get_likely_next_roles(
                    self.config.author, previous_role, top_k=3
                )
                if likely_roles:
                    # Sample from likely roles weighted by probability
                    roles, probs = zip(*likely_roles)
                    role = random.choices(roles, weights=probs, k=1)[0]
                else:
                    role = source_role
            else:
                role = source_role

            # First sentence should usually be CLAIM in intro paragraphs
            if i == 0 and paragraph_position == "INTRO":
                role = RhetoricalRole.CLAIM

            plans.append({
                "type": sent_type,
                "role": role,
                "relation": source_relation,
            })

            previous_role = role

        return plans

    def _transfer_sentence(
        self,
        source: str,
        sentence_type: SentenceType,
        rhetorical_role: RhetoricalRole,
        logical_relation: LogicalRelation,
        technical_terms: Set[str],
        used_words: Set[str],
        position: int,
        total: int
    ) -> Optional[FilledSentence]:
        """Transfer a single sentence.

        Args:
            source: Source sentence.
            sentence_type: Target type.
            rhetorical_role: Target role.
            logical_relation: Relation to previous.
            technical_terms: Terms to preserve.
            used_words: Already used words.
            position: Position in paragraph.
            total: Total sentences.

        Returns:
            FilledSentence or None if failed.
        """
        logger.debug(f"--- Transferring sentence {position+1}/{total} ---")
        logger.debug(f"Source: '{source[:60]}...'")
        logger.debug(f"Target: type={sentence_type.value}, role={rhetorical_role.value}, relation={logical_relation.value}")

        # Extract proposition from source
        proposition = Proposition.from_sentence(source)
        proposition.technical_terms.update(technical_terms)

        logger.debug(f"Proposition: subj='{proposition.subject[:30] if proposition.subject else 'None'}', "
                    f"pred='{proposition.predicate}', obj='{proposition.object[:30] if proposition.object else 'None'}'")

        # Find matching templates
        templates = self.store.query_templates(
            query_text=source,
            author=self.config.author,
            sentence_type=sentence_type,
            rhetorical_role=rhetorical_role,
            n_results=self.config.n_template_candidates,
        )
        logger.debug(f"Found {len(templates)} templates with type+role filter")

        if not templates:
            # Fallback: query without role constraint
            logger.debug("Falling back to type-only filter")
            templates = self.store.query_templates(
                query_text=source,
                author=self.config.author,
                sentence_type=sentence_type,
                n_results=self.config.n_template_candidates,
            )
            logger.debug(f"Found {len(templates)} templates with type-only filter")

        if not templates:
            # Last resort: any template from author
            logger.debug("Falling back to author-only filter")
            templates = self.store.query_templates(
                query_text=source,
                author=self.config.author,
                n_results=self.config.n_template_candidates,
            )
            logger.debug(f"Found {len(templates)} templates with author-only filter")

        if not templates:
            logger.warning(f"No templates found for: {source[:50]}...")
            # Create filled sentence directly from source
            return FilledSentence(
                template=SentenceTemplate(
                    id="direct",
                    skeleton=source,
                    pos_pattern="",
                    word_count=len(source.split()),
                    complexity_score=3.0,
                    clause_count=1,
                    sentence_type=sentence_type,
                    rhetorical_role=rhetorical_role,
                    logical_relation=logical_relation,
                    original_text=source,
                ),
                text=source,
                filled_slots=[],
                propositions_used=[proposition],
                technical_terms_preserved=technical_terms,
            )

        # Find best matching template
        best_template = self.matcher.find_best_template(
            proposition, templates, self.config.min_template_match_score
        )

        if not best_template:
            best_template = templates[0]
            logger.debug(f"No template matched threshold, using first: '{best_template.skeleton[:40]}...'")
        else:
            logger.debug(f"Best matching template: '{best_template.skeleton[:40]}...'")

        logger.debug(f"Template details: slots={len(best_template.slots)}, "
                    f"words={best_template.word_count}, complexity={best_template.complexity_score:.1f}")

        # Fill template
        filled = self.filler.fill_template(
            best_template, [proposition], used_words
        )

        if not filled:
            logger.warning(f"Failed to fill template for: {source[:50]}...")
            return None

        logger.debug(f"Filled result: '{filled.text[:60]}...'")

        # Validate and repair
        sentence_terms = technical_terms & {
            w for w in source.split()
            if w.lower() in {t.lower() for t in technical_terms}
        }

        validation = self.sentence_validator.validate(
            filled.text, sentence_terms, position, total
        )

        if not validation.is_valid:
            logger.debug(f"Sentence validation issues: {[i.type for i in validation.issues]}")
            repaired_text, success = self.sentence_repairer.repair(
                filled.text,
                validation,
                sentence_terms,
                self.config.max_sentence_repair_iterations,
            )
            if success or repaired_text != filled.text:
                filled = FilledSentence(
                    template=filled.template,
                    text=repaired_text,
                    filled_slots=filled.filled_slots,
                    propositions_used=filled.propositions_used,
                    technical_terms_preserved=filled.technical_terms_preserved,
                )

        return filled

    def _enforce_rhythm(self, paragraph: str) -> str:
        """Enforce rhythm pattern to match author's style.

        Args:
            paragraph: Paragraph to adjust.

        Returns:
            Rhythm-adjusted paragraph.
        """
        sentences = split_into_sentences(paragraph)
        if len(sentences) < 3 or not self.corpus_stats.rhythm_patterns:
            return paragraph

        # Find best matching target pattern
        target_pattern = None
        for pattern in self.corpus_stats.rhythm_patterns:
            if len(pattern) == len(sentences):
                target_pattern = pattern
                break

        if not target_pattern:
            return paragraph

        # Compute current pattern
        lengths = [len(s.split()) for s in sentences]
        mean_length = sum(lengths) / len(lengths)

        if mean_length == 0:
            return paragraph

        # Adjust each sentence
        adjusted = []
        for sent, target_ratio in zip(sentences, target_pattern):
            target_length = int(target_ratio * mean_length)
            current_length = len(sent.split())

            if abs(current_length - target_length) <= 2:
                # Close enough
                adjusted.append(sent)
            elif current_length < target_length:
                # Need to extend
                adjusted.append(self._extend_sentence(sent, target_length))
            else:
                # Need to shorten
                adjusted.append(self._shorten_sentence(sent, target_length))

        return " ".join(adjusted)

    def _extend_sentence(self, sentence: str, target: int) -> str:
        """Extend sentence to target length."""
        doc = self.nlp(sentence)
        words = [t.text for t in doc]
        current = len([w for w in words if w.strip()])

        if current >= target or not self.vocabulary:
            return sentence

        # Add modifiers before nouns
        modifiers = list(self.vocabulary.modifiers.keys())[:5]
        if not modifiers:
            return sentence

        result = []
        mod_idx = 0

        for token in doc:
            # Add modifier before noun if no adjective child
            if (token.pos_ == "NOUN" and
                current < target and
                not any(c.pos_ == "ADJ" for c in token.children)):
                result.append(modifiers[mod_idx % len(modifiers)])
                mod_idx += 1
                current += 1

            result.append(token.text)

        return " ".join(result)

    def _shorten_sentence(self, sentence: str, target: int) -> str:
        """Shorten sentence to target length."""
        doc = self.nlp(sentence)
        current = len([t for t in doc if not t.is_punct and not t.is_space])

        if current <= target:
            return sentence

        # Remove optional modifiers
        essential_deps = {"ROOT", "nsubj", "nsubjpass", "dobj", "pobj", "attr"}
        words = []

        for token in doc:
            if token.dep_ in essential_deps or token.is_punct:
                words.append(token.text)
            elif token.pos_ not in ("ADJ", "ADV"):
                words.append(token.text)
            elif len(words) < target:
                words.append(token.text)

        result = " ".join(words)
        # Clean up spacing
        result = result.replace(" ,", ",").replace(" .", ".")
        return result


@dataclass
class SemanticTransferResult:
    """Result of semantic style transfer."""
    original: str
    transferred: str
    prompt_used: str
    style_delta: float
    verification_passed: bool
    issues: List[str] = field(default_factory=list)
    repairs_applied: int = 0


class SemanticStylePipeline:
    """Pipeline for semantic-preserving style transfer.

    Uses:
    - spaCy-based vocabulary classification (transferable vs topic-specific)
    - Discourse context (document/paragraph/sentence roles)
    - Template scaffolding (structure hints for LLM)
    - Style fingerprint (statistical voice signature)
    - Burrows' Delta verification
    """

    def __init__(
        self,
        corpus_paragraphs: List[str],
        author: str,
        config: Optional[TransferConfig] = None,
        llm_generator: Optional[Callable[[str], str]] = None,
    ):
        """Initialize semantic style pipeline.

        Args:
            corpus_paragraphs: Author's corpus for style extraction.
            author: Author name.
            config: Transfer configuration.
            llm_generator: Function that takes prompt and returns generated text.
        """
        self._nlp = None
        self.author = author
        self.config = config or TransferConfig(author=author)
        self.llm_generator = llm_generator or self.config.llm_generator
        self.corpus_paragraphs = corpus_paragraphs

        logger.info(f"Initializing SemanticStylePipeline for {author}")
        logger.info(f"Corpus: {len(corpus_paragraphs)} paragraphs")

        # Extract transferable style using spaCy
        logger.info("Classifying vocabulary (transferable vs topic-specific)...")
        self.vocab_classifier = VocabularyClassifier()
        self.classified_vocab = self.vocab_classifier.classify_corpus(corpus_paragraphs)

        logger.info(f"  Transferable transitions: {list(self.classified_vocab.transitions.keys())[:8]}")
        logger.info(f"  General verbs: {list(self.classified_vocab.general_verbs.keys())[:8]}")
        logger.info(f"  Topic-specific (excluded): {list(self.classified_vocab.topic_specific)[:8]}")

        # Extract style fingerprint
        logger.info("Extracting style fingerprint...")
        self.fingerprint_extractor = StyleFingerprintExtractor()
        self.fingerprint = self.fingerprint_extractor.extract(corpus_paragraphs)

        logger.info(f"  Sentence length: {self.fingerprint.sentence_length_mean:.1f} ± {self.fingerprint.sentence_length_std:.1f}")
        logger.info(f"  Burstiness: {self.fingerprint.burstiness_mean:.3f}")

        # Initialize verifier
        self.verifier = StyleVerifier(self.fingerprint, threshold=self.config.delta_threshold)

        # Template abstractor for structural hints
        self.template_abstractor = TemplateAbstractor()

        # Discourse analyzer
        self.discourse_analyzer = DiscourseAnalyzer()

        # Technical term extractor
        self.term_extractor = TechnicalTermExtractor()

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def build_sentence_prompt(
        self,
        source_sentence: str,
        sentence_context: SentenceContext,
        paragraph_context: ParagraphContext,
        technical_terms: Set[str],
        template_hint: Optional[str] = None,
    ) -> str:
        """Build a context-aware prompt for sentence generation.

        Args:
            source_sentence: Source sentence to transfer.
            sentence_context: Sentence-level context.
            paragraph_context: Paragraph-level context.
            technical_terms: Terms to preserve exactly.
            template_hint: Optional structural template.

        Returns:
            Complete prompt for LLM.
        """
        lines = []

        # Task description
        lines.append(f"=== TASK: STYLE TRANSFER ===")
        lines.append(f"Rewrite the source content in the style of {self.author}.")
        lines.append("")
        lines.append("CRITICAL RULES:")
        lines.append("1. PRESERVE the meaning and key terms from the source")
        lines.append("2. APPLY the author's sentence structure and rhythm")
        lines.append("3. USE the author's transition words and general vocabulary")
        lines.append("4. DO NOT add concepts or terms not present in the source")
        lines.append("")

        # Source content
        lines.append("=== SOURCE CONTENT (preserve this meaning) ===")
        lines.append(f"Original: \"{source_sentence}\"")

        # Extract and show proposition
        prop = Proposition.from_sentence(source_sentence)
        lines.append("")
        lines.append("Core proposition:")
        if prop.subject:
            lines.append(f"  Subject: {prop.subject}")
        lines.append(f"  Predicate: {prop.predicate}")
        if prop.object:
            lines.append(f"  Object: {prop.object}")

        # Required terms
        source_nouns = self._extract_key_terms(source_sentence)
        required_terms = source_nouns | technical_terms
        if required_terms:
            lines.append(f"\nTerms that MUST appear: {', '.join(list(required_terms)[:10])}")

        if technical_terms:
            lines.append(f"Technical terms (preserve exactly): {', '.join(list(technical_terms)[:5])}")
        lines.append("")

        # Template structure hint
        if template_hint:
            lines.append("=== SENTENCE STRUCTURE (from author) ===")
            lines.append(f"Pattern hint: {template_hint}")
            lines.append("")

        # Author's transferable style
        lines.append("=== AUTHOR'S STYLE (apply these) ===")
        target_min = max(5, int(self.fingerprint.sentence_length_mean - self.fingerprint.sentence_length_std))
        target_max = int(self.fingerprint.sentence_length_mean + self.fingerprint.sentence_length_std)
        lines.append(f"Sentence length: {target_min}-{target_max} words")

        # Transitions (transferable)
        transitions = list(self.classified_vocab.transitions.keys())[:10]
        if transitions:
            lines.append(f"Transition preferences: {', '.join(transitions)}")

        # General vocabulary (transferable)
        lines.append("")
        lines.append("General vocabulary preferences (use where natural):")
        verbs = list(self.classified_vocab.general_verbs.keys())[:12]
        if verbs:
            lines.append(f"  Verbs: {', '.join(verbs)}")
        adjs = list(self.classified_vocab.general_adjectives.keys())[:12]
        if adjs:
            lines.append(f"  Adjectives: {', '.join(adjs)}")
        nouns = list(self.classified_vocab.general_nouns.keys())[:12]
        if nouns:
            lines.append(f"  Abstract nouns: {', '.join(nouns)}")
        lines.append("")

        # Discourse context
        lines.append("=== DISCOURSE CONTEXT ===")
        lines.append(f"Position: {paragraph_context.document_position.value} paragraph")
        lines.append(f"Sentence {sentence_context.position + 1} of {sentence_context.total_in_paragraph}")
        lines.append(f"Sentence function: {sentence_context.role.value}")

        if sentence_context.discourse_relation != DiscourseRelation.NONE:
            lines.append(f"Relation to previous: {sentence_context.discourse_relation.value}")
            markers = DISCOURSE_MARKERS.get(sentence_context.discourse_relation, [])
            if markers:
                lines.append(f"Consider starting with: {', '.join(markers[:4])}")

        if sentence_context.previous_topic:
            lines.append(f"Previous topic: {sentence_context.previous_topic[:60]}...")
        lines.append("")

        # Final instruction
        lines.append("=== OUTPUT ===")
        lines.append("Generate ONE sentence that expresses the source meaning in the author's style.")
        lines.append("Write ONLY the sentence:")

        return "\n".join(lines)

    def transfer_sentence(
        self,
        source_sentence: str,
        sentence_context: SentenceContext,
        paragraph_context: ParagraphContext,
        technical_terms: Set[str],
    ) -> SemanticTransferResult:
        """Transfer a single sentence with semantic preservation.

        Args:
            source_sentence: Source sentence.
            sentence_context: Sentence-level context.
            paragraph_context: Paragraph-level context.
            technical_terms: Terms to preserve.

        Returns:
            SemanticTransferResult.
        """
        # Build prompt
        prompt = self.build_sentence_prompt(
            source_sentence,
            sentence_context,
            paragraph_context,
            technical_terms,
        )

        # Generate with LLM if available
        if self.llm_generator:
            generated = self.llm_generator(prompt)
        else:
            # Fallback: return source (for testing without LLM)
            logger.warning("No LLM generator configured, returning source")
            generated = source_sentence

        # Verify style match
        verification = self.verifier.verify(generated)

        # Attempt repairs if needed
        repairs = 0
        while not verification.is_acceptable and repairs < self.config.max_sentence_repair_iterations:
            repairs += 1
            logger.debug(f"Style verification failed (delta={verification.delta_score:.2f}), repair attempt {repairs}")

            # Build repair prompt
            repair_prompt = self._build_repair_prompt(
                prompt, generated, verification
            )

            if self.llm_generator:
                generated = self.llm_generator(repair_prompt)
                verification = self.verifier.verify(generated)
            else:
                break

        return SemanticTransferResult(
            original=source_sentence,
            transferred=generated,
            prompt_used=prompt,
            style_delta=verification.delta_score,
            verification_passed=verification.is_acceptable,
            issues=verification.issues,
            repairs_applied=repairs,
        )

    def transfer_paragraph(
        self,
        paragraph: str,
        paragraph_position: DocumentPosition = DocumentPosition.BODY,
        paragraph_index: int = 0,
        total_paragraphs: int = 1,
    ) -> TransferResult:
        """Transfer a paragraph with discourse awareness.

        Args:
            paragraph: Input paragraph.
            paragraph_position: Position in document.
            paragraph_index: Index of paragraph.
            total_paragraphs: Total paragraphs in document.

        Returns:
            TransferResult.
        """
        logger.info(f"Transferring paragraph ({paragraph_position.value})")

        # Extract technical terms
        technical_terms = self.term_extractor.extract(paragraph)
        logger.debug(f"Technical terms: {technical_terms}")

        # Split into sentences
        sentences = split_into_sentences(paragraph)
        if not sentences:
            return TransferResult(
                original=paragraph,
                transferred=paragraph,
                sentences=[],
                technical_terms_preserved=technical_terms,
                validation_passed=False,
                burstiness=0.0,
                corpus_similarity=0.0,
            )

        # Create paragraph context
        para_ctx = ParagraphContext(
            position=paragraph_index,
            total_in_document=total_paragraphs,
            document_position=paragraph_position,
            role=self._infer_paragraph_role(paragraph, paragraph_position),
            discourse_relation=DiscourseRelation.NONE if paragraph_index == 0 else DiscourseRelation.ELABORATION,
        )

        # Transfer each sentence
        transferred_sentences = []
        filled_sentences = []
        previous_topic = None

        for i, sent in enumerate(sentences):
            # Create sentence context
            sent_ctx = SentenceContext(
                position=i,
                total_in_paragraph=len(sentences),
                role=self._infer_sentence_role(i, len(sentences)),
                discourse_relation=self._infer_discourse_relation(sent, i),
                reference_type=None,
                previous_topic=previous_topic,
            )

            # Transfer sentence
            result = self.transfer_sentence(
                sent, sent_ctx, para_ctx, technical_terms
            )

            transferred_sentences.append(result.transferred)
            previous_topic = sent[:50]

            # Create FilledSentence for compatibility
            filled_sentences.append(FilledSentence(
                template=SentenceTemplate(
                    id=f"semantic_{i}",
                    skeleton="",
                    pos_pattern="",
                    word_count=len(result.transferred.split()),
                    complexity_score=3.0,
                    clause_count=1,
                    sentence_type=SentenceType.BODY,
                    rhetorical_role=RhetoricalRole.CLAIM,
                    logical_relation=LogicalRelation.NONE,
                    original_text=result.original,
                ),
                text=result.transferred,
                filled_slots=[],
                propositions_used=[],
                technical_terms_preserved=technical_terms,
            ))

        # Combine
        transferred_text = " ".join(transferred_sentences)

        # Calculate burstiness
        burstiness = calculate_burstiness(transferred_sentences) if len(transferred_sentences) > 1 else 0

        # Verify paragraph
        para_verification = self.verifier.verify(transferred_text)

        return TransferResult(
            original=paragraph,
            transferred=transferred_text,
            sentences=filled_sentences,
            technical_terms_preserved=technical_terms,
            validation_passed=para_verification.is_acceptable,
            burstiness=burstiness,
            corpus_similarity=1.0 - (para_verification.delta_score / 5.0),  # Normalize
        )

    def transfer_document(
        self,
        paragraphs: List[str],
        paragraph_callback: Optional[Callable] = None,
    ) -> List[TransferResult]:
        """Transfer an entire document.

        Args:
            paragraphs: List of paragraphs.
            paragraph_callback: Optional callback after each paragraph.

        Returns:
            List of TransferResults.
        """
        results = []

        for i, para in enumerate(paragraphs):
            # Determine position
            if i == 0:
                position = DocumentPosition.INTRO
            elif i == len(paragraphs) - 1:
                position = DocumentPosition.CONCLUSION
            else:
                position = DocumentPosition.BODY

            logger.info(f"Transferring paragraph {i+1}/{len(paragraphs)} ({position.value})")

            result = self.transfer_paragraph(
                para,
                paragraph_position=position,
                paragraph_index=i,
                total_paragraphs=len(paragraphs),
            )
            results.append(result)

            if paragraph_callback:
                paragraph_callback(i, result)

        return results

    def _extract_key_terms(self, sentence: str) -> Set[str]:
        """Extract key terms from sentence that must be preserved."""
        doc = self.nlp(sentence)
        terms = set()

        for token in doc:
            # Keep nouns and main verbs
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                terms.add(token.text.lower())
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                terms.add(token.lemma_.lower())

        return terms

    def _infer_paragraph_role(
        self,
        paragraph: str,
        position: DocumentPosition,
    ) -> ParagraphRole:
        """Infer paragraph role from content and position."""
        if position == DocumentPosition.INTRO:
            return ParagraphRole.THESIS
        if position == DocumentPosition.CONCLUSION:
            return ParagraphRole.CONCLUSION
        return ParagraphRole.ELABORATION

    def _infer_sentence_role(self, position: int, total: int) -> SentenceRole:
        """Infer sentence role from position."""
        if position == 0:
            return SentenceRole.TOPIC
        if position == total - 1:
            return SentenceRole.CONCLUSION
        return SentenceRole.SUPPORT

    def _infer_discourse_relation(self, sentence: str, position: int) -> DiscourseRelation:
        """Infer discourse relation from content."""
        if position == 0:
            return DiscourseRelation.NONE

        sent_lower = sentence.lower()

        # Check for explicit markers
        for relation, markers in DISCOURSE_MARKERS.items():
            for marker in markers:
                if sent_lower.startswith(marker) or sent_lower.startswith(marker + ","):
                    return relation

        return DiscourseRelation.ELABORATION

    def _build_repair_prompt(
        self,
        original_prompt: str,
        failed_text: str,
        verification,
    ) -> str:
        """Build a repair prompt for failed generation."""
        lines = []
        lines.append("The previous generation did not match the author's style.")
        lines.append("")
        lines.append(f"Previous attempt: \"{failed_text}\"")
        lines.append(f"Style delta: {verification.delta_score:.2f} (too high)")
        lines.append("")
        lines.append("Issues:")
        for issue in verification.issues:
            lines.append(f"  - {issue}")
        lines.append("")
        lines.append("Please regenerate with these corrections.")
        lines.append("")
        lines.append(original_prompt)

        return "\n".join(lines)

    def get_style_summary(self) -> Dict:
        """Get a summary of extracted style for debugging."""
        return {
            "author": self.author,
            "sentence_length": {
                "mean": round(self.fingerprint.sentence_length_mean, 1),
                "std": round(self.fingerprint.sentence_length_std, 1),
            },
            "burstiness": round(self.fingerprint.burstiness_mean, 3),
            "transferable": {
                "transitions": list(self.classified_vocab.transitions.keys())[:10],
                "verbs": list(self.classified_vocab.general_verbs.keys())[:10],
                "adjectives": list(self.classified_vocab.general_adjectives.keys())[:10],
                "nouns": list(self.classified_vocab.general_nouns.keys())[:10],
            },
            "excluded_topic_specific": list(self.classified_vocab.topic_specific)[:15],
            "excluded_entities": list(self.classified_vocab.named_entities)[:10],
        }


def create_transfer_pipeline(
    corpus_paragraphs: List[str],
    author: str,
    persist_path: Optional[str] = None,
    use_semantic: bool = True,
    llm_generator: Optional[Callable[[str], str]] = None,
) -> "TemplateBasedTransfer | SemanticStylePipeline":
    """Create and initialize a transfer pipeline.

    Args:
        corpus_paragraphs: Author's corpus paragraphs.
        author: Author name.
        persist_path: Optional path to persist ChromaDB.
        use_semantic: Use new semantic-preserving pipeline.
        llm_generator: LLM generation function for semantic pipeline.

    Returns:
        Initialized pipeline (semantic or template-based).
    """
    if use_semantic:
        logger.info("Creating SemanticStylePipeline")
        config = TransferConfig(author=author, llm_generator=llm_generator)
        return SemanticStylePipeline(
            corpus_paragraphs=corpus_paragraphs,
            author=author,
            config=config,
            llm_generator=llm_generator,
        )

    # Legacy template-based pipeline
    from ..config import ChromaDBConfig

    # Create store
    config = ChromaDBConfig(persist_path=persist_path) if persist_path else ChromaDBConfig()
    store = TemplateStore(config)

    # Index corpus
    logger.info(f"Indexing {len(corpus_paragraphs)} paragraphs for {author}")
    store.index_corpus(corpus_paragraphs, author)

    # Create pipeline
    transfer_config = TransferConfig(author=author)
    return TemplateBasedTransfer(store, transfer_config)
