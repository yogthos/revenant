"""Proposition-based evolutionary repair for semantic preservation.

This module implements a sophisticated repair system that:
1. Maps output sentences to source propositions
2. Identifies missing/fabricated content at proposition level
3. Generates targeted repairs using semantic relationships
4. Evolves candidates through multi-generation selection

The key insight: Instead of regenerating entire paragraphs, we make
surgical fixes to specific sentences based on proposition analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Literal
from enum import Enum
from difflib import SequenceMatcher
import re
import json
from pathlib import Path

from .semantic_graph import (
    SemanticGraph,
    SemanticGraphBuilder,
    SemanticGraphComparator,
    PropositionNode,
    RelationEdge,
    RelationType,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _load_validation_config() -> dict:
    """Load validation config from config.json."""
    config_paths = [
        Path("config.json"),
        Path(__file__).parent.parent.parent / "config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    full_config = json.load(f)
                return full_config.get("validation", {})
            except Exception:
                pass

    return {}  # No config found, use defaults


# Load config once at module level
_CONFIG = _load_validation_config()


class SentenceClassification(Enum):
    """Classification of an output sentence."""
    GROUNDED = "grounded"      # All content maps to source
    PARTIAL = "partial"        # Some content maps, some doesn't
    HALLUCINATED = "hallucinated"  # No content maps to source
    SHORT = "short"            # Too short to analyze


@dataclass
class PropositionMatch:
    """A match between a source proposition and output content."""
    source_prop: PropositionNode
    match_score: float  # 0-1 similarity
    matched_in_sentence_idx: Optional[int] = None  # Which output sentence
    is_exact: bool = False


@dataclass
class SentenceAnalysis:
    """Analysis of a single output sentence against source graph."""
    sentence: str
    index: int

    # Propositions from source that this sentence expresses
    matched_propositions: List[PropositionMatch] = field(default_factory=list)

    # Content in this sentence not found in source
    novel_content: List[str] = field(default_factory=list)

    # Classification
    classification: SentenceClassification = SentenceClassification.SHORT

    # Scores
    grounding_score: float = 0.0  # How well grounded in source
    novelty_score: float = 0.0    # How much novel content

    def is_problematic(self) -> bool:
        """Check if this sentence needs repair."""
        return self.classification in (
            SentenceClassification.HALLUCINATED,
            SentenceClassification.PARTIAL
        ) and self.novelty_score > 0.3


@dataclass
class OutputAnalysis:
    """Complete analysis of output against source semantic graph."""
    # Source information
    source_graph: SemanticGraph
    source_text: str

    # Output breakdown
    output_sentences: List[str]
    sentence_analyses: List[SentenceAnalysis]

    # Proposition tracking
    covered_propositions: Set[str] = field(default_factory=set)  # Prop IDs found
    missing_propositions: List[PropositionNode] = field(default_factory=list)

    # Overall scores
    coverage_score: float = 0.0      # Fraction of source props covered
    hallucination_count: int = 0     # Number of hallucinated sentences
    grounded_ratio: float = 0.0      # Fraction of sentences grounded

    def get_repair_priority(self) -> List[Tuple[int, str]]:
        """Get sentences needing repair, ordered by priority.

        Returns:
            List of (sentence_index, reason) tuples.
        """
        repairs = []

        # Priority 1: Hallucinated sentences (remove or replace)
        for sa in self.sentence_analyses:
            if sa.classification == SentenceClassification.HALLUCINATED:
                repairs.append((sa.index, "hallucinated"))

        # Priority 2: Partial sentences with high novelty
        for sa in self.sentence_analyses:
            if sa.classification == SentenceClassification.PARTIAL and sa.novelty_score > 0.5:
                repairs.append((sa.index, "high_novelty"))

        return repairs


@dataclass
class RepairAction:
    """A specific repair action to take."""
    action_type: Literal["remove", "replace", "fix", "add"]
    target_sentence_idx: Optional[int]  # Which sentence to modify

    # What to add/preserve
    propositions_to_include: List[PropositionNode] = field(default_factory=list)
    context_propositions: List[PropositionNode] = field(default_factory=list)

    # What to remove
    content_to_remove: List[str] = field(default_factory=list)

    # Insertion point for "add" actions
    insert_after_idx: Optional[int] = None

    def to_prompt_hint(self, author: str) -> str:
        """Generate prompt hint for this repair."""
        hints = []

        if self.action_type == "remove":
            return ""  # No generation needed

        if self.propositions_to_include:
            props_text = "; ".join(p.text for p in self.propositions_to_include[:3])
            hints.append(f"MUST EXPRESS: {props_text}")

        if self.content_to_remove:
            remove_text = ", ".join(self.content_to_remove[:3])
            hints.append(f"DO NOT INCLUDE: {remove_text}")

        if self.context_propositions:
            context_text = "; ".join(p.text for p in self.context_propositions[:2])
            hints.append(f"CONTEXT: {context_text}")

        hints.append(f"Write in {author}'s voice. Include ONLY the specified information.")

        return "[" + " | ".join(hints) + "]"


@dataclass
class RepairCandidate:
    """A candidate repair with quality scores."""
    text: str

    # Quality metrics
    proposition_coverage: float  # Does it contain required props?
    hallucination_score: float   # Does it avoid fabrication?
    fluency_score: float         # Does it read naturally?
    length_ratio: float          # Is it appropriate length?
    novelty_score: float = 1.0   # Does it avoid repeating existing sentences?

    @property
    def overall_score(self) -> float:
        """Combined quality score."""
        # Novelty is critical - repeating content is useless
        if self.novelty_score < 0.5:
            return 0.0  # Reject highly repetitive candidates
        return (
            0.30 * self.proposition_coverage +
            0.30 * self.hallucination_score +
            0.10 * self.fluency_score +
            0.10 * min(1.0, self.length_ratio) +
            0.20 * self.novelty_score  # Reward novel (non-repetitive) content
        )


class PropositionAnalyzer:
    """Analyzes output text against source semantic graph at proposition level."""

    def __init__(
        self,
        match_threshold: Optional[float] = None,
        novelty_threshold: Optional[float] = None,
    ):
        # Use config values with fallback defaults
        self.match_threshold = match_threshold or _CONFIG.get("proposition_match_threshold", 0.4)
        self.novelty_threshold = novelty_threshold or _CONFIG.get("novelty_threshold", 0.5)
        self._nlp = None
        self._builder = None

    @property
    def nlp(self):
        if self._nlp is None:
            from ..utils.nlp import get_nlp
            self._nlp = get_nlp()
        return self._nlp

    @property
    def builder(self):
        if self._builder is None:
            self._builder = SemanticGraphBuilder()
        return self._builder

    def analyze(
        self,
        source_text: str,
        source_graph: SemanticGraph,
        output_text: str,
    ) -> OutputAnalysis:
        """Analyze output against source at proposition level.

        Args:
            source_text: Original source text.
            source_graph: Semantic graph of source.
            output_text: Generated output text.

        Returns:
            OutputAnalysis with detailed breakdown.
        """
        # Split output into sentences
        doc = self.nlp(output_text)
        output_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not output_sentences:
            return OutputAnalysis(
                source_graph=source_graph,
                source_text=source_text,
                output_sentences=[],
                sentence_analyses=[],
            )

        # Get source proposition content words for matching
        source_prop_words = self._extract_proposition_words(source_graph)
        all_source_words = set()
        for words in source_prop_words.values():
            all_source_words.update(words)

        # Analyze each output sentence
        sentence_analyses = []
        covered_props = set()
        hallucination_count = 0

        for idx, sent in enumerate(output_sentences):
            analysis = self._analyze_sentence(
                sent, idx, source_graph, source_prop_words, all_source_words
            )
            sentence_analyses.append(analysis)

            # Track covered propositions
            for match in analysis.matched_propositions:
                covered_props.add(match.source_prop.id)

            if analysis.classification == SentenceClassification.HALLUCINATED:
                hallucination_count += 1

        # Find missing propositions
        missing = [
            node for node in source_graph.nodes
            if node.id not in covered_props
        ]

        # Calculate scores
        coverage = len(covered_props) / len(source_graph.nodes) if source_graph.nodes else 1.0
        grounded_count = sum(
            1 for sa in sentence_analyses
            if sa.classification in (SentenceClassification.GROUNDED, SentenceClassification.SHORT)
        )
        grounded_ratio = grounded_count / len(sentence_analyses) if sentence_analyses else 1.0

        return OutputAnalysis(
            source_graph=source_graph,
            source_text=source_text,
            output_sentences=output_sentences,
            sentence_analyses=sentence_analyses,
            covered_propositions=covered_props,
            missing_propositions=missing,
            coverage_score=coverage,
            hallucination_count=hallucination_count,
            grounded_ratio=grounded_ratio,
        )

    def _extract_proposition_words(
        self,
        graph: SemanticGraph,
    ) -> Dict[str, Set[str]]:
        """Extract content words for each proposition."""
        prop_words = {}

        for node in graph.nodes:
            doc = self.nlp(node.text)
            words = set()
            for token in doc:
                if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM'} and not token.is_stop:
                    words.add(token.lemma_.lower())
            prop_words[node.id] = words

        return prop_words

    def _analyze_sentence(
        self,
        sentence: str,
        index: int,
        source_graph: SemanticGraph,
        source_prop_words: Dict[str, Set[str]],
        all_source_words: Set[str],
    ) -> SentenceAnalysis:
        """Analyze a single sentence against source propositions."""
        # Skip short sentences
        words = sentence.split()
        if len(words) < 5:
            return SentenceAnalysis(
                sentence=sentence,
                index=index,
                classification=SentenceClassification.SHORT,
                grounding_score=1.0,
            )

        # Extract content words from sentence
        doc = self.nlp(sentence)
        sent_words = set()
        for token in doc:
            if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM'} and not token.is_stop:
                sent_words.add(token.lemma_.lower())

        if not sent_words:
            return SentenceAnalysis(
                sentence=sentence,
                index=index,
                classification=SentenceClassification.SHORT,
                grounding_score=1.0,
            )

        # Match against each source proposition
        matches = []
        for node in source_graph.nodes:
            prop_words = source_prop_words.get(node.id, set())
            if not prop_words:
                continue

            # Calculate overlap
            overlap = sent_words & prop_words
            if not overlap:
                continue

            precision = len(overlap) / len(sent_words)
            recall = len(overlap) / len(prop_words)

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0

            if f1 >= self.match_threshold:
                matches.append(PropositionMatch(
                    source_prop=node,
                    match_score=f1,
                    matched_in_sentence_idx=index,
                    is_exact=f1 > 0.8,
                ))

        # Calculate novelty (words not in any source proposition)
        novel_words = sent_words - all_source_words
        novelty_score = len(novel_words) / len(sent_words) if sent_words else 0

        # Determine classification
        if not matches and novelty_score > self.novelty_threshold:
            classification = SentenceClassification.HALLUCINATED
        elif matches and novelty_score < 0.3:
            classification = SentenceClassification.GROUNDED
        elif matches:
            classification = SentenceClassification.PARTIAL
        else:
            # No matches but low novelty - might just be different wording
            classification = SentenceClassification.PARTIAL

        # Grounding score based on matches and novelty
        match_score = max((m.match_score for m in matches), default=0)
        grounding_score = match_score * (1 - novelty_score * 0.5)

        return SentenceAnalysis(
            sentence=sentence,
            index=index,
            matched_propositions=matches,
            novel_content=list(novel_words)[:10],
            classification=classification,
            grounding_score=grounding_score,
            novelty_score=novelty_score,
        )


class EvolutionaryRepairer:
    """Repairs output using evolutionary multi-candidate selection.

    Strategy:
    1. Analyze output to find proposition gaps and hallucinations
    2. Generate repair plan based on semantic relationships
    3. For each repair, generate multiple candidates
    4. Select best candidate using multi-criteria scoring
    5. Iterate until convergence or max iterations
    """

    def __init__(
        self,
        generator,  # LoRAStyleGenerator
        author: str,
        coverage_threshold: Optional[float] = None,
        candidates_per_repair: Optional[int] = None,
        max_iterations: int = 2,  # Reduced default - fewer iterations = less repetition
    ):
        self.generator = generator
        self.author = author
        # Use config values with fallback defaults
        self.coverage_threshold = coverage_threshold or _CONFIG.get("coverage_threshold", 0.85)
        self.candidates_per_repair = candidates_per_repair or _CONFIG.get("candidates_per_repair", 2)
        self.max_iterations = _CONFIG.get("max_repair_iterations", max_iterations)

        self._analyzer = None
        self._builder = None

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = PropositionAnalyzer()
        return self._analyzer

    @property
    def builder(self):
        if self._builder is None:
            self._builder = SemanticGraphBuilder()
        return self._builder

    def repair(
        self,
        source_text: str,
        output_text: str,
        source_graph: Optional[SemanticGraph] = None,
    ) -> Tuple[str, float, OutputAnalysis]:
        """Repair output to match source semantics.

        Args:
            source_text: Original source text.
            output_text: Generated output to repair.
            source_graph: Pre-built semantic graph (optional).

        Returns:
            Tuple of (repaired_text, coverage_score, final_analysis).
        """
        # Build source graph if not provided
        if source_graph is None:
            source_graph = self.builder.build_from_text(source_text)

        if not source_graph.nodes:
            # No propositions to validate against
            return output_text, 1.0, OutputAnalysis(
                source_graph=source_graph,
                source_text=source_text,
                output_sentences=[output_text],
                sentence_analyses=[],
            )

        # Track source length for expansion limiting
        source_word_count = len(source_text.split())
        max_allowed_words = int(source_word_count * 1.5)  # Max 50% expansion

        best_output = output_text
        best_score = 0.0
        best_analysis = None
        previous_quality = 0.0  # Track quality across iterations

        for iteration in range(self.max_iterations):
            # Analyze current state
            analysis = self.analyzer.analyze(source_text, source_graph, output_text)

            # Compute overall quality including coherence
            output_word_count = len(output_text.split())
            repetition_ratio = self._compute_repetition_ratio(output_text)
            expansion_ratio = output_word_count / source_word_count if source_word_count > 0 else 1.0

            # Quality penalizes both hallucinations and repetition
            quality = (
                analysis.coverage_score * 0.4 +
                analysis.grounded_ratio * 0.3 +
                (1.0 - repetition_ratio) * 0.2 +  # Penalize repetition
                min(1.0, 1.0 / max(expansion_ratio, 0.5)) * 0.1  # Penalize expansion
            )

            logger.info(
                f"Repair iteration {iteration + 1}/{self.max_iterations}: "
                f"coverage={analysis.coverage_score:.2f}, "
                f"hallucinations={analysis.hallucination_count}, "
                f"grounded={analysis.grounded_ratio:.2f}, "
                f"repetition={repetition_ratio:.2f}, "
                f"expansion={expansion_ratio:.2f}"
            )

            # Track best by combined quality (not just coverage)
            if quality > best_score:
                best_score = quality
                best_output = output_text
                best_analysis = analysis

            # Check if quality is degrading
            if iteration > 0 and quality < previous_quality - 0.1:
                logger.warning(
                    f"Quality degraded ({previous_quality:.2f} -> {quality:.2f}), "
                    f"stopping repairs"
                )
                break

            previous_quality = quality

            # Check for over-expansion
            if output_word_count > max_allowed_words:
                logger.warning(
                    f"Output too long ({output_word_count} > {max_allowed_words} words), "
                    f"stopping repairs"
                )
                break

            # Check for excessive repetition
            if repetition_ratio > 0.4:
                logger.warning(
                    f"Excessive repetition ({repetition_ratio:.0%}), stopping repairs"
                )
                break

            # Check convergence
            if (analysis.coverage_score >= self.coverage_threshold and
                analysis.hallucination_count == 0):
                logger.info("Converged: coverage threshold met with no hallucinations")
                return output_text, analysis.coverage_score, analysis

            # Generate repair plan
            repair_actions = self._plan_repairs(analysis, source_graph)

            if not repair_actions:
                logger.info("No repair actions identified")
                break

            # Apply repairs
            output_text = self._apply_repairs(
                output_text,
                analysis,
                repair_actions,
                source_graph,
            )

        # Return best seen
        if best_analysis is None:
            best_analysis = self.analyzer.analyze(source_text, source_graph, best_output)

        return best_output, best_score, best_analysis

    def _plan_repairs(
        self,
        analysis: OutputAnalysis,
        source_graph: SemanticGraph,
    ) -> List[RepairAction]:
        """Generate repair plan based on analysis.

        Strategy: Prioritize COHERENCE over coverage.
        - Remove hallucinations (they hurt quality)
        - Replace only if we can substitute missing content
        - DON'T add new sentences (causes repetition bloat)
        """
        actions = []

        # Action 1: Remove hallucinated sentences
        # Only replace if we have unhandled missing propositions
        for sa in analysis.sentence_analyses:
            if sa.classification == SentenceClassification.HALLUCINATED:
                # Check if we have missing propositions to substitute
                if analysis.missing_propositions and len(actions) == 0:
                    # Replace FIRST hallucination with missing content (once only)
                    actions.append(RepairAction(
                        action_type="replace",
                        target_sentence_idx=sa.index,
                        propositions_to_include=analysis.missing_propositions[:1],  # Just one
                        context_propositions=self._get_adjacent_props(
                            analysis.missing_propositions[0],
                            source_graph
                        ),
                    ))
                else:
                    # Just remove subsequent hallucinations
                    actions.append(RepairAction(
                        action_type="remove",
                        target_sentence_idx=sa.index,
                    ))

        # Action 2: Fix partial sentences with high novelty (only first one)
        for sa in analysis.sentence_analyses:
            if sa.is_problematic() and sa.classification == SentenceClassification.PARTIAL:
                # Only fix if we haven't already done too many repairs
                if len([a for a in actions if a.action_type != "remove"]) < 1:
                    matched_props = [m.source_prop for m in sa.matched_propositions]
                    actions.append(RepairAction(
                        action_type="fix",
                        target_sentence_idx=sa.index,
                        propositions_to_include=matched_props,
                        content_to_remove=sa.novel_content[:5],
                    ))

        # NO Action 3: We intentionally skip adding new propositions
        # Adding content leads to repetition and bloat
        # Better to have slightly lower coverage but coherent text

        return actions

    def _get_adjacent_props(
        self,
        prop: PropositionNode,
        graph: SemanticGraph,
    ) -> List[PropositionNode]:
        """Get propositions connected to this one in the graph."""
        adjacent = []

        for edge in graph.edges:
            if edge.source_id == prop.id:
                node = graph.get_node(edge.target_id)
                if node:
                    adjacent.append(node)
            elif edge.target_id == prop.id:
                node = graph.get_node(edge.source_id)
                if node:
                    adjacent.append(node)

        return adjacent[:3]  # Limit context

    def _find_insertion_point(
        self,
        prop: PropositionNode,
        analysis: OutputAnalysis,
        graph: SemanticGraph,
    ) -> int:
        """Find best position to insert a missing proposition."""
        # Look for sentences that contain related propositions
        adjacent = self._get_adjacent_props(prop, graph)
        adjacent_ids = {p.id for p in adjacent}

        for sa in analysis.sentence_analyses:
            for match in sa.matched_propositions:
                if match.source_prop.id in adjacent_ids:
                    return sa.index

        # Default: end of text
        return len(analysis.output_sentences) - 1

    def _apply_repairs(
        self,
        output_text: str,
        analysis: OutputAnalysis,
        actions: List[RepairAction],
        source_graph: SemanticGraph,
    ) -> str:
        """Apply repair actions to output text."""
        sentences = list(analysis.output_sentences)

        # Sort actions by sentence index (reverse for safe removal)
        indexed_actions = [(a.target_sentence_idx or a.insert_after_idx or 0, a) for a in actions]
        indexed_actions.sort(key=lambda x: x[0], reverse=True)

        for _, action in indexed_actions:
            if action.action_type == "remove":
                if 0 <= action.target_sentence_idx < len(sentences):
                    sentences.pop(action.target_sentence_idx)

            elif action.action_type == "replace":
                candidate = self._generate_best_candidate(action, sentences, analysis)
                if candidate and 0 <= action.target_sentence_idx < len(sentences):
                    sentences[action.target_sentence_idx] = candidate.text

            elif action.action_type == "fix":
                candidate = self._generate_best_candidate(action, sentences, analysis)
                if candidate and 0 <= action.target_sentence_idx < len(sentences):
                    sentences[action.target_sentence_idx] = candidate.text

            elif action.action_type == "add":
                candidate = self._generate_best_candidate(action, sentences, analysis)
                if candidate:
                    insert_idx = (action.insert_after_idx or 0) + 1
                    insert_idx = min(insert_idx, len(sentences))
                    sentences.insert(insert_idx, candidate.text)

        # Deduplicate sentences to prevent repetition
        sentences = self._deduplicate_sentences(sentences)

        return " ".join(sentences)

    def _deduplicate_sentences(
        self,
        sentences: List[str],
        similarity_threshold: float = 0.7,
    ) -> List[str]:
        """Remove duplicate or near-duplicate sentences.

        Uses sequence matching to identify similar sentences and keeps
        only the first occurrence.
        """
        if len(sentences) <= 1:
            return sentences

        unique = []
        seen_normalized = set()

        for sent in sentences:
            # Normalize for comparison
            normalized = sent.lower().strip()
            normalized = re.sub(r'[^\w\s]', '', normalized)
            normalized = re.sub(r'\s+', ' ', normalized)

            # Skip exact duplicates
            if normalized in seen_normalized:
                logger.debug(f"Removed exact duplicate: '{sent[:50]}...'")
                continue

            # Check similarity with already kept sentences
            is_duplicate = False
            for kept in unique:
                kept_normalized = kept.lower().strip()
                kept_normalized = re.sub(r'[^\w\s]', '', kept_normalized)
                kept_normalized = re.sub(r'\s+', ' ', kept_normalized)

                ratio = SequenceMatcher(None, normalized, kept_normalized).ratio()
                if ratio >= similarity_threshold:
                    logger.debug(f"Removed similar sentence (ratio={ratio:.2f}): '{sent[:50]}...'")
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(sent)
                seen_normalized.add(normalized)

        if len(unique) < len(sentences):
            logger.info(f"Deduplication removed {len(sentences) - len(unique)} repetitive sentences")

        return unique

    def _compute_repetition_ratio(self, text: str) -> float:
        """Compute how much of the text is repetitive.

        Returns ratio from 0.0 (no repetition) to 1.0 (all repetition).
        Uses sentence similarity to detect near-duplicates.
        """
        nlp = self.analyzer.nlp
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if len(sentences) <= 1:
            return 0.0

        # Normalize sentences
        normalized = []
        for sent in sentences:
            norm = sent.lower()
            norm = re.sub(r'[^\w\s]', '', norm)
            norm = re.sub(r'\s+', ' ', norm).strip()
            normalized.append(norm)

        # Count pairs with high similarity
        repetitive_count = 0
        total_pairs = 0

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                total_pairs += 1
                ratio = SequenceMatcher(None, normalized[i], normalized[j]).ratio()
                if ratio >= 0.6:  # Consider as repetitive if 60%+ similar
                    repetitive_count += 1

        if total_pairs == 0:
            return 0.0

        return repetitive_count / total_pairs

    def _generate_best_candidate(
        self,
        action: RepairAction,
        current_sentences: List[str],
        analysis: OutputAnalysis,
    ) -> Optional[RepairCandidate]:
        """Generate multiple candidates and select the best."""
        if not self.generator:
            return None

        candidates = []
        temperatures = [0.2, 0.35, 0.5][:self.candidates_per_repair]

        # Build prompt
        prompt_hint = action.to_prompt_hint(self.author)

        # Get content to generate
        if action.propositions_to_include:
            base_content = " ".join(p.text for p in action.propositions_to_include)
        else:
            return None

        target_words = len(base_content.split())

        for temp in temperatures:
            try:
                content = f"{prompt_hint}\n\n{base_content}"

                generated = self.generator.generate(
                    content=content,
                    author=self.author,
                    target_words=target_words,
                    temperature=temp,
                )

                if generated and len(generated.split()) >= 5:
                    candidate = self._score_candidate(
                        generated,
                        action,
                        analysis,
                        current_sentences,
                    )
                    candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Candidate generation failed: {e}")

        if not candidates:
            return None

        # Select best
        return max(candidates, key=lambda c: c.overall_score)

    def _score_candidate(
        self,
        text: str,
        action: RepairAction,
        analysis: OutputAnalysis,
        current_sentences: List[str],
    ) -> RepairCandidate:
        """Score a candidate repair."""
        # Check proposition coverage
        required_props = action.propositions_to_include
        required_words = set()
        for prop in required_props:
            doc = self.analyzer.nlp(prop.text)
            for token in doc:
                if token.pos_ in {'NOUN', 'VERB', 'PROPN'} and not token.is_stop:
                    required_words.add(token.lemma_.lower())

        candidate_doc = self.analyzer.nlp(text)
        candidate_words = set()
        for token in candidate_doc:
            if token.pos_ in {'NOUN', 'VERB', 'PROPN'} and not token.is_stop:
                candidate_words.add(token.lemma_.lower())

        if required_words:
            coverage = len(required_words & candidate_words) / len(required_words)
        else:
            coverage = 1.0

        # Check for hallucination (novel content not in source)
        all_source_words = set()
        for prop in analysis.source_graph.nodes:
            doc = self.analyzer.nlp(prop.text)
            for token in doc:
                if not token.is_stop:
                    all_source_words.add(token.lemma_.lower())

        novel = candidate_words - all_source_words
        hallucination_score = 1.0 - (len(novel) / len(candidate_words) if candidate_words else 0)

        # Simple fluency check (has complete sentence structure)
        has_period = text.strip().endswith(('.', '!', '?'))
        word_count = len(text.split())
        fluency = 0.8 if has_period and 5 <= word_count <= 50 else 0.5

        # Length ratio
        target = sum(len(p.text.split()) for p in required_props)
        length_ratio = word_count / target if target > 0 else 1.0

        # Novelty score: how different is this from existing sentences?
        novelty_score = self._compute_novelty_score(text, current_sentences)

        return RepairCandidate(
            text=text,
            proposition_coverage=coverage,
            hallucination_score=hallucination_score,
            fluency_score=fluency,
            length_ratio=length_ratio,
            novelty_score=novelty_score,
        )

    def _compute_novelty_score(
        self,
        candidate: str,
        existing_sentences: List[str],
    ) -> float:
        """Compute how novel (non-repetitive) a candidate is.

        Returns 1.0 if completely novel, 0.0 if exact duplicate.
        """
        if not existing_sentences:
            return 1.0

        candidate_normalized = candidate.lower().strip()
        candidate_normalized = re.sub(r'[^\w\s]', '', candidate_normalized)
        candidate_normalized = re.sub(r'\s+', ' ', candidate_normalized)

        max_similarity = 0.0
        for sent in existing_sentences:
            sent_normalized = sent.lower().strip()
            sent_normalized = re.sub(r'[^\w\s]', '', sent_normalized)
            sent_normalized = re.sub(r'\s+', ' ', sent_normalized)

            ratio = SequenceMatcher(None, candidate_normalized, sent_normalized).ratio()
            max_similarity = max(max_similarity, ratio)

        # Novelty is inverse of similarity
        return 1.0 - max_similarity


def repair_with_propositions(
    source_text: str,
    output_text: str,
    generator,
    author: str,
    source_graph: Optional[SemanticGraph] = None,
    coverage_threshold: float = 0.85,
    max_iterations: int = 3,
) -> Tuple[str, float, OutputAnalysis]:
    """Convenience function for proposition-based repair.

    Args:
        source_text: Original source text.
        output_text: Generated output to repair.
        generator: LoRAStyleGenerator instance.
        author: Author name for style.
        source_graph: Pre-built semantic graph (optional).
        coverage_threshold: Min coverage to accept.
        max_iterations: Max repair iterations.

    Returns:
        Tuple of (repaired_text, coverage_score, analysis).
    """
    repairer = EvolutionaryRepairer(
        generator=generator,
        author=author,
        coverage_threshold=coverage_threshold,
        max_iterations=max_iterations,
    )

    return repairer.repair(source_text, output_text, source_graph)
