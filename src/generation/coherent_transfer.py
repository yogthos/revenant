"""Coherent paragraph-level style transfer.

This module implements a holistic approach to style transfer that:
1. Plans paragraph structure from semantic relationships
2. Generates whole paragraphs (not individual sentences)
3. Validates coherence at paragraph level
4. Regenerates (never patches) when validation fails

Key insight: Coherent paragraphs must be generated as wholes, not assembled
from independently generated parts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, deque
from difflib import SequenceMatcher
import re
import numpy as np

from .lora_generator import LoRAStyleGenerator
from ..validation.semantic_graph import (
    SemanticGraph,
    SemanticGraphBuilder,
    PropositionNode,
    RelationType,
)
from ..utils.nlp import get_nlp
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParagraphPlan:
    """Plan for generating a coherent paragraph.

    The plan orders propositions based on semantic relationships and
    identifies transition points where the relationship type changes.
    """
    # Ordered propositions to include
    propositions: List[PropositionNode]

    # Transition markers: index -> transition hint for that position
    transitions: Dict[int, str] = field(default_factory=dict)

    # Target length in words
    target_words: int = 100

    # Critical entities that must appear
    required_entities: List[str] = field(default_factory=list)

    # Content to avoid (from failed attempts)
    avoid_content: List[str] = field(default_factory=list)

    def to_prompt_bullets(self) -> str:
        """Convert to numbered bullet points for generation prompt."""
        lines = []
        for i, prop in enumerate(self.propositions):
            transition = self.transitions.get(i, "")
            if transition:
                lines.append(f"{i+1}. [{transition}] {prop.text}")
            else:
                lines.append(f"{i+1}. {prop.text}")
        return "\n".join(lines)

    def simplified(self) -> 'ParagraphPlan':
        """Return a simplified plan with fewer propositions."""
        if len(self.propositions) <= 3:
            return self

        # Keep first, last, and every other middle proposition
        simplified_props = [self.propositions[0]]
        for i in range(1, len(self.propositions) - 1, 2):
            simplified_props.append(self.propositions[i])
        simplified_props.append(self.propositions[-1])

        return ParagraphPlan(
            propositions=simplified_props,
            transitions={},  # Regenerate transitions
            target_words=self.target_words,
            required_entities=self.required_entities,
            avoid_content=self.avoid_content,
        )


@dataclass
class CoherenceScore:
    """Detailed coherence metrics for a paragraph."""
    topic_continuity: float  # How well sentences connect topically (0-1)
    length_variance: float   # Sentence length variety - some is good (0-1)
    repetition_score: float  # 1 = no repetition, 0 = all repetition
    flow_score: float        # How well sentences transition (0-1)

    @property
    def overall(self) -> float:
        """Combined coherence score."""
        return (
            0.35 * self.topic_continuity +
            0.15 * self.length_variance +
            0.30 * self.repetition_score +
            0.20 * self.flow_score
        )

    def __str__(self) -> str:
        return (f"CoherenceScore(topic={self.topic_continuity:.2f}, "
                f"variance={self.length_variance:.2f}, "
                f"repetition={self.repetition_score:.2f}, "
                f"flow={self.flow_score:.2f}, "
                f"overall={self.overall:.2f})")


@dataclass
class ValidationResult:
    """Result of holistic paragraph validation."""
    proposition_coverage: float
    grounding_score: float
    coherence: CoherenceScore
    length_ratio: float
    entity_coverage: float

    @property
    def overall_score(self) -> float:
        """Combined quality score."""
        return (
            0.30 * self.proposition_coverage +
            0.25 * self.grounding_score +
            0.30 * self.coherence.overall +
            0.15 * self.entity_coverage
        )

    @property
    def is_valid(self) -> bool:
        """Whether the paragraph meets quality thresholds."""
        return (
            self.proposition_coverage >= 0.70 and
            self.grounding_score >= 0.65 and
            self.coherence.overall >= 0.55 and
            0.6 <= self.length_ratio <= 1.6 and
            self.coherence.repetition_score >= 0.7
        )

    def get_issues(self) -> List[str]:
        """Get list of validation issues."""
        issues = []
        if self.proposition_coverage < 0.70:
            issues.append(f"Low proposition coverage: {self.proposition_coverage:.0%}")
        if self.grounding_score < 0.65:
            issues.append(f"Low grounding (hallucinations): {self.grounding_score:.0%}")
        if self.coherence.overall < 0.55:
            issues.append(f"Low coherence: {self.coherence.overall:.0%}")
        if self.coherence.repetition_score < 0.7:
            issues.append(f"Too much repetition: {1-self.coherence.repetition_score:.0%}")
        if not (0.6 <= self.length_ratio <= 1.6):
            issues.append(f"Length issue: {self.length_ratio:.0%} of target")
        return issues


class ParagraphPlanner:
    """Plans paragraph structure from semantic graph.

    The planner analyzes relationships between propositions to determine
    the optimal ordering for coherent text generation.
    """

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def create_plan(
        self,
        graph: SemanticGraph,
        source_text: str,
    ) -> ParagraphPlan:
        """Create a generation plan from the semantic graph.

        Args:
            graph: Semantic graph with propositions and relationships
            source_text: Original source text (for length calculation)

        Returns:
            ParagraphPlan with ordered propositions and transition markers
        """
        if not graph.nodes:
            return ParagraphPlan(
                propositions=[],
                target_words=len(source_text.split()),
            )

        # Step 1: Build dependency edges from relationships
        dependencies = self._build_dependencies(graph)

        # Step 2: Topological sort with cycle handling
        ordered = self._topological_sort(graph, dependencies)

        # Step 3: Identify transition points
        transitions = self._identify_transitions(ordered, graph)

        # Step 4: Extract required entities
        entities = self._extract_entities(graph)

        # Step 5: Calculate target words
        target_words = len(source_text.split())

        logger.debug(f"Created plan with {len(ordered)} propositions, "
                    f"{len(transitions)} transitions, target={target_words} words")

        return ParagraphPlan(
            propositions=ordered,
            transitions=transitions,
            target_words=target_words,
            required_entities=entities,
        )

    def _build_dependencies(self, graph: SemanticGraph) -> Dict[str, Set[str]]:
        """Build dependency graph from relationships.

        Returns dict where deps[A] = {B, C} means B and C must come before A.
        """
        deps = defaultdict(set)

        for edge in graph.edges:
            rel_type = edge.relation  # Note: attribute is 'relation', not 'relation_type'
            source_id = edge.source_id
            target_id = edge.target_id

            # Determine ordering based on relationship type
            if rel_type in [RelationType.CAUSE, RelationType.SEQUENCE]:
                # Source causes/precedes target -> source before target
                deps[target_id].add(source_id)
            elif rel_type == RelationType.ELABORATION:
                # Target elaborates source -> source before target
                deps[target_id].add(source_id)
            elif rel_type == RelationType.EXAMPLE:
                # Target is example of source -> source before target
                deps[target_id].add(source_id)
            elif rel_type == RelationType.SUPPORT:
                # Support can go either way, but evidence often comes first
                # We'll put supporting evidence before the claim
                deps[target_id].add(source_id)
            elif rel_type == RelationType.CONDITION:
                # Condition before result
                deps[target_id].add(source_id)
            # CONTRAST, RESTATEMENT: no strict ordering constraint

        return deps

    def _topological_sort(
        self,
        graph: SemanticGraph,
        deps: Dict[str, Set[str]],
    ) -> List[PropositionNode]:
        """Topologically sort nodes, handling cycles gracefully.

        Uses Kahn's algorithm. If cycles exist, breaks them by including
        remaining nodes in their original order.
        """
        # Calculate in-degree for each node
        in_degree = defaultdict(int)
        for node in graph.nodes:
            in_degree[node.id] = len(deps.get(node.id, set()))

        # Initialize queue with nodes that have no dependencies
        queue = deque()
        for node in graph.nodes:
            if in_degree[node.id] == 0:
                queue.append(node.id)

        result = []
        processed = set()

        while queue:
            node_id = queue.popleft()
            node = graph.get_node(node_id)
            if node:
                result.append(node)
                processed.add(node_id)

            # Decrease in-degree of nodes that depend on this one
            for other_id, dep_set in deps.items():
                if node_id in dep_set and other_id not in processed:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] <= 0 and other_id not in processed:
                        queue.append(other_id)

        # Handle any remaining nodes (cycles or disconnected)
        remaining = [n for n in graph.nodes if n.id not in processed]
        if remaining:
            logger.debug(f"Added {len(remaining)} nodes not in dependency order")
            result.extend(remaining)

        return result

    def _identify_transitions(
        self,
        ordered: List[PropositionNode],
        graph: SemanticGraph,
    ) -> Dict[int, str]:
        """Identify transition types between consecutive propositions.

        Returns dict mapping position index to transition hint.
        """
        transitions = {}

        # Build relationship lookup
        relationships = {}
        for edge in graph.edges:
            key = (edge.source_id, edge.target_id)
            relationships[key] = edge.relation
            # Also add reverse for undirected relationships
            if edge.relation in [RelationType.CONTRAST, RelationType.RESTATEMENT]:
                relationships[(edge.target_id, edge.source_id)] = edge.relation

        for i in range(1, len(ordered)):
            prev_id = ordered[i-1].id
            curr_id = ordered[i].id

            # Check for direct relationship
            rel = relationships.get((prev_id, curr_id))
            if not rel:
                rel = relationships.get((curr_id, prev_id))

            if rel:
                hint = self._relation_to_hint(rel)
                if hint:
                    transitions[i] = hint

        return transitions

    def _relation_to_hint(self, rel_type: RelationType) -> str:
        """Convert relationship type to generation hint."""
        hints = {
            RelationType.CAUSE: "consequently",
            RelationType.SEQUENCE: "then",
            RelationType.ELABORATION: "specifically",
            RelationType.CONTRAST: "however",
            RelationType.EXAMPLE: "for instance",
            RelationType.SUPPORT: "indeed",
            RelationType.CONDITION: "therefore",
            RelationType.RESTATEMENT: "in other words",
        }
        return hints.get(rel_type, "")

    def _extract_entities(self, graph: SemanticGraph) -> List[str]:
        """Extract named entities that must be preserved."""
        entities = set()

        for node in graph.nodes:
            doc = self.nlp(node.text)
            for ent in doc.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE', 'DATE', 'MONEY',
                                  'PERCENT', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}:
                    entities.add(ent.text)

        return list(entities)


class CoherentGenerator:
    """Generates coherent paragraphs from plans.

    Uses structured prompts that include ordered propositions and
    transition hints to guide the model toward coherent output.
    """

    def __init__(self, lora_generator: LoRAStyleGenerator):
        self.lora_generator = lora_generator

    def generate(
        self,
        plan: ParagraphPlan,
        author: str,
        temperature: float = 0.3,
    ) -> str:
        """Generate a coherent paragraph following the plan.

        Args:
            plan: Paragraph plan with ordered propositions
            author: Author name for style
            temperature: Generation temperature

        Returns:
            Generated paragraph text
        """
        if not plan.propositions:
            return ""

        prompt = self._build_prompt(plan, author)

        output = self.lora_generator.generate(
            content=prompt,
            author=author,
            target_words=plan.target_words,
            temperature=temperature,
        )

        return output or ""

    def _build_prompt(self, plan: ParagraphPlan, author: str) -> str:
        """Build a structured prompt for coherent generation.

        The prompt explicitly lists propositions in order with transition
        hints, emphasizing the need for coherent flow.
        """
        lines = [
            f"Transform into a coherent paragraph in {author}'s voice.",
            "",
            "Include these points in this order:",
        ]

        # Add numbered propositions with transition hints
        lines.append(plan.to_prompt_bullets())

        lines.extend([
            "",
            "Requirements:",
            f"- Write approximately {plan.target_words} words",
            "- Each sentence must flow naturally to the next",
            "- Include all numbered points in order",
            "- Do not repeat the same idea twice",
        ])

        if plan.required_entities:
            entities_str = ", ".join(plan.required_entities[:5])
            lines.append(f"- Must mention: {entities_str}")

        if plan.avoid_content:
            avoid_str = "; ".join(plan.avoid_content[:3])
            lines.append(f"- Avoid: {avoid_str}")

        lines.append("")
        lines.append("Write the paragraph:")

        return "\n".join(lines)


class HolisticValidator:
    """Validates whole paragraphs, not individual sentences.

    Checks:
    1. Proposition coverage - Are all planned propositions present?
    2. Grounding - Is content grounded in source (no hallucinations)?
    3. Coherence - Do sentences flow together naturally?
    4. Length - Is output appropriate length?
    5. Entities - Are required entities present?
    """

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def validate(
        self,
        source: str,
        output: str,
        plan: ParagraphPlan,
    ) -> ValidationResult:
        """Validate a generated paragraph holistically.

        Args:
            source: Original source text
            output: Generated paragraph
            plan: The plan used for generation

        Returns:
            ValidationResult with detailed metrics
        """
        # Check proposition coverage
        coverage = self._check_coverage(output, plan.propositions)

        # Check grounding (no hallucinations)
        grounding = self._check_grounding(source, output)

        # Check coherence
        coherence = self._check_coherence(output)

        # Check length
        target = plan.target_words
        actual = len(output.split())
        length_ratio = actual / target if target > 0 else 1.0

        # Check entities
        entity_coverage = self._check_entities(output, plan.required_entities)

        result = ValidationResult(
            proposition_coverage=coverage,
            grounding_score=grounding,
            coherence=coherence,
            length_ratio=length_ratio,
            entity_coverage=entity_coverage,
        )

        logger.debug(f"Validation: coverage={coverage:.2f}, grounding={grounding:.2f}, "
                    f"coherence={coherence.overall:.2f}, length_ratio={length_ratio:.2f}")

        return result

    def _check_coverage(
        self,
        output: str,
        propositions: List[PropositionNode],
    ) -> float:
        """Check what fraction of propositions are covered in output."""
        if not propositions:
            return 1.0

        output_doc = self.nlp(output.lower())
        output_words = set()
        for token in output_doc:
            if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN'} and not token.is_stop:
                output_words.add(token.lemma_.lower())

        covered = 0
        for prop in propositions:
            prop_doc = self.nlp(prop.text.lower())
            prop_words = set()
            for token in prop_doc:
                if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN'} and not token.is_stop:
                    prop_words.add(token.lemma_.lower())

            if prop_words:
                overlap = len(prop_words & output_words) / len(prop_words)
                if overlap >= 0.5:  # Consider covered if 50%+ words present
                    covered += 1

        return covered / len(propositions)

    def _check_grounding(self, source: str, output: str) -> float:
        """Check that output is grounded in source (no hallucinations)."""
        source_doc = self.nlp(source.lower())
        output_doc = self.nlp(output.lower())

        # Get content words from source
        source_words = set()
        for token in source_doc:
            if not token.is_stop and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}:
                source_words.add(token.lemma_.lower())

        # Get content words from output
        output_words = set()
        for token in output_doc:
            if not token.is_stop and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}:
                output_words.add(token.lemma_.lower())

        if not output_words:
            return 1.0

        # What fraction of output words are in source?
        grounded = output_words & source_words
        return len(grounded) / len(output_words)

    def _check_coherence(self, text: str) -> CoherenceScore:
        """Check paragraph coherence holistically."""
        doc = self.nlp(text)
        sentences = [sent for sent in doc.sents if sent.text.strip()]

        if len(sentences) <= 1:
            return CoherenceScore(
                topic_continuity=1.0,
                length_variance=0.5,
                repetition_score=1.0,
                flow_score=0.8,
            )

        # Topic continuity: content word overlap between adjacent sentences
        continuity_scores = []
        for i in range(len(sentences) - 1):
            words1 = self._get_content_words(sentences[i])
            words2 = self._get_content_words(sentences[i + 1])
            if words1 and words2:
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                # Scale up - even small overlap is good
                continuity_scores.append(min(1.0, overlap * 2.5))
            else:
                continuity_scores.append(0.5)
        topic_continuity = np.mean(continuity_scores) if continuity_scores else 0.5

        # Length variance: some variation is natural and good
        lengths = [len(list(sent)) for sent in sentences]
        if len(lengths) > 1 and np.mean(lengths) > 0:
            cv = np.std(lengths) / np.mean(lengths)
            # CV of 0.3-0.5 is ideal (natural variation)
            if 0.2 <= cv <= 0.6:
                length_variance = 1.0
            elif cv < 0.2:
                length_variance = cv / 0.2  # Too uniform
            else:
                length_variance = max(0.5, 1.0 - (cv - 0.6) / 0.4)  # Too variable
        else:
            length_variance = 0.5

        # Repetition: check for duplicate/similar sentences
        repetition_score = self._check_repetition(sentences)

        # Flow: check for appropriate transitions
        flow_score = self._check_flow(sentences)

        return CoherenceScore(
            topic_continuity=topic_continuity,
            length_variance=length_variance,
            repetition_score=repetition_score,
            flow_score=flow_score,
        )

    def _get_content_words(self, sent) -> Set[str]:
        """Get content word lemmas from a sentence."""
        words = set()
        for token in sent:
            if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'PROPN'} and not token.is_stop:
                words.add(token.lemma_.lower())
        return words

    def _check_repetition(self, sentences) -> float:
        """Check for sentence-level repetition."""
        if len(sentences) <= 1:
            return 1.0

        # Normalize sentences for comparison
        normalized = []
        for sent in sentences:
            norm = sent.text.lower()
            norm = re.sub(r'[^\w\s]', '', norm)
            norm = re.sub(r'\s+', ' ', norm).strip()
            normalized.append(norm)

        # Check for exact duplicates
        unique_ratio = len(set(normalized)) / len(normalized)

        # Check for high similarity between any pair
        similarity_penalty = 0
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                if normalized[i] and normalized[j]:
                    sim = SequenceMatcher(None, normalized[i], normalized[j]).ratio()
                    if sim > 0.6:
                        # Penalize based on similarity level
                        similarity_penalty += (sim - 0.6) * 0.5

        return max(0, min(1.0, unique_ratio - similarity_penalty))

    def _check_flow(self, sentences) -> float:
        """Check for appropriate use of transitions."""
        if len(sentences) <= 1:
            return 1.0

        connectors = {
            'however', 'therefore', 'thus', 'moreover', 'furthermore',
            'yet', 'but', 'and', 'so', 'because', 'although', 'while',
            'meanwhile', 'consequently', 'nevertheless', 'indeed',
            'specifically', 'additionally', 'finally', 'first', 'then',
        }

        # Count sentences that start with or contain connectors
        connector_count = 0
        for sent in sentences:
            words = sent.text.lower().split()
            if words and (words[0] in connectors or
                         any(w in connectors for w in words[:3])):
                connector_count += 1

        connector_ratio = connector_count / len(sentences)

        # Ideal ratio is around 0.2-0.4 (some but not all sentences)
        if 0.15 <= connector_ratio <= 0.5:
            return 1.0
        elif connector_ratio < 0.15:
            return 0.7 + connector_ratio * 2  # Slightly penalize few connectors
        else:
            return max(0.6, 1.0 - (connector_ratio - 0.5))  # Penalize too many

    def _check_entities(self, output: str, required: List[str]) -> float:
        """Check coverage of required entities."""
        if not required:
            return 1.0

        output_lower = output.lower()
        found = sum(1 for ent in required if ent.lower() in output_lower)
        return found / len(required)


class RegenerativeRefiner:
    """Refines output through regeneration, never patching.

    When validation fails, adjusts the plan and regenerates the entire
    paragraph. This maintains coherence (patches create seams).
    """

    def __init__(
        self,
        generator: CoherentGenerator,
        validator: HolisticValidator,
        max_attempts: int = 3,
    ):
        self.generator = generator
        self.validator = validator
        self.max_attempts = max_attempts

    def refine(
        self,
        source: str,
        plan: ParagraphPlan,
        author: str,
    ) -> Tuple[str, ValidationResult]:
        """Generate and refine until valid or max attempts reached.

        Args:
            source: Original source text
            plan: Initial paragraph plan
            author: Author name for style

        Returns:
            Tuple of (best_output, validation_result)
        """
        best_output = ""
        best_score = -1.0
        best_validation = None

        current_plan = plan

        for attempt in range(self.max_attempts):
            # Adjust temperature: start low, increase if stuck
            temperature = 0.25 + attempt * 0.12

            # Generate
            output = self.generator.generate(current_plan, author, temperature)

            if not output:
                continue

            # Validate
            validation = self.validator.validate(source, output, plan)

            # Calculate overall score
            score = validation.overall_score

            logger.info(
                f"Attempt {attempt + 1}/{self.max_attempts}: "
                f"score={score:.2f}, valid={validation.is_valid}"
            )

            # Track best
            if score > best_score:
                best_output = output
                best_score = score
                best_validation = validation

            # Check if valid
            if validation.is_valid:
                logger.info("Generation successful - validation passed")
                return output, validation

            # Log issues
            issues = validation.get_issues()
            if issues:
                logger.debug(f"Issues: {', '.join(issues)}")

            # Adjust plan for next attempt
            current_plan = self._adjust_plan(current_plan, validation, output)

        logger.info(f"Returning best attempt with score={best_score:.2f}")
        return best_output, best_validation

    def _adjust_plan(
        self,
        plan: ParagraphPlan,
        validation: ValidationResult,
        failed_output: str,
    ) -> ParagraphPlan:
        """Adjust plan based on validation failures."""
        new_plan = ParagraphPlan(
            propositions=list(plan.propositions),
            transitions=dict(plan.transitions),
            target_words=plan.target_words,
            required_entities=list(plan.required_entities),
            avoid_content=list(plan.avoid_content),
        )

        # If too much repetition, explicitly tell model to avoid it
        if validation.coherence.repetition_score < 0.7:
            # Find repeated phrases in failed output
            new_plan.avoid_content.append("repetitive phrases")
            new_plan.avoid_content.append("restating the same idea")

        # If coherence is very low, simplify the plan
        if validation.coherence.overall < 0.5 and len(plan.propositions) > 4:
            logger.debug("Simplifying plan due to low coherence")
            new_plan = new_plan.simplified()

        # If length is off, adjust target
        if validation.length_ratio < 0.7:
            new_plan.target_words = int(plan.target_words * 0.85)
        elif validation.length_ratio > 1.4:
            new_plan.target_words = int(plan.target_words * 1.15)

        return new_plan


class CoherentStyleTransfer:
    """Main class for coherent paragraph-level style transfer.

    Coordinates planning, generation, validation, and refinement
    to produce coherent, styled paragraphs.
    """

    def __init__(
        self,
        lora_generator: LoRAStyleGenerator,
        author: str,
    ):
        self.lora_generator = lora_generator
        self.author = author

        self.planner = ParagraphPlanner()
        self.generator = CoherentGenerator(lora_generator)
        self.validator = HolisticValidator()
        self.refiner = RegenerativeRefiner(self.generator, self.validator)

        self._graph_builder = None

    @property
    def graph_builder(self):
        if self._graph_builder is None:
            self._graph_builder = SemanticGraphBuilder()
        return self._graph_builder

    def transfer(
        self,
        source_text: str,
        source_graph: Optional[SemanticGraph] = None,
    ) -> Tuple[str, bool]:
        """Transfer style while maintaining coherence.

        Args:
            source_text: Original paragraph text
            source_graph: Pre-built semantic graph (optional)

        Returns:
            Tuple of (styled_text, validation_passed)
        """
        # Build semantic graph if not provided
        if source_graph is None:
            source_graph = self.graph_builder.build_from_text(source_text)

        if not source_graph.nodes:
            # No propositions - just do simple generation
            logger.debug("No propositions found, using simple generation")
            output = self.lora_generator.generate(
                content=source_text,
                author=self.author,
                target_words=len(source_text.split()),
            )
            return output or source_text, True

        # Create plan from semantic graph
        plan = self.planner.create_plan(source_graph, source_text)

        logger.info(f"Created plan with {len(plan.propositions)} propositions")

        # Generate and refine
        output, validation = self.refiner.refine(
            source=source_text,
            plan=plan,
            author=self.author,
        )

        # Final deduplication pass
        output = self._deduplicate(output)

        return output, validation.is_valid if validation else False

    def _deduplicate(self, text: str) -> str:
        """Remove any remaining duplicate sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        unique = []
        seen = set()

        for sent in sentences:
            if not sent.strip():
                continue

            norm = sent.lower()
            norm = re.sub(r'[^\w\s]', '', norm)
            norm = re.sub(r'\s+', ' ', norm).strip()

            if norm in seen:
                continue

            is_dup = False
            for kept in unique:
                kept_norm = kept.lower()
                kept_norm = re.sub(r'[^\w\s]', '', kept_norm)
                kept_norm = re.sub(r'\s+', ' ', kept_norm).strip()
                if SequenceMatcher(None, norm, kept_norm).ratio() >= 0.7:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(sent)
                seen.add(norm)

        return ' '.join(unique)


def transfer_paragraph_coherently(
    source_text: str,
    lora_generator: LoRAStyleGenerator,
    author: str,
    source_graph: Optional[SemanticGraph] = None,
) -> Tuple[str, bool]:
    """Convenience function for coherent style transfer.

    Args:
        source_text: Original paragraph
        lora_generator: LoRA generator instance
        author: Author name
        source_graph: Optional pre-built semantic graph

    Returns:
        Tuple of (styled_text, validation_passed)
    """
    transferer = CoherentStyleTransfer(lora_generator, author)
    return transferer.transfer(source_text, source_graph)
