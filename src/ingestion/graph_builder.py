"""Semantic graph construction from propositions."""

from typing import List, Optional

from ..models.graph import (
    PropositionNode,
    RelationshipEdge,
    RelationshipType,
    SemanticGraph,
    DocumentGraph,
    ParagraphRole,
    RhetoricalIntent,
)
from ..corpus.preprocessor import ProcessedDocument, ProcessedParagraph
from ..llm.provider import LLMProvider
from ..utils.logging import get_logger
from .proposition_extractor import PropositionExtractor
from .relationship_detector import RelationshipDetector

logger = get_logger(__name__)


class SemanticGraphBuilder:
    """Builds semantic graphs from text.

    Orchestrates proposition extraction and relationship detection
    to create a complete semantic graph for a paragraph.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        use_llm_relationships: bool = True
    ):
        """Initialize graph builder.

        Args:
            llm_provider: Optional LLM provider for enhanced detection.
            use_llm_relationships: Whether to use LLM for relationship detection.
        """
        self.llm_provider = llm_provider
        self.prop_extractor = PropositionExtractor()
        self.rel_detector = RelationshipDetector(
            llm_provider=llm_provider,
            use_llm_fallback=use_llm_relationships
        )

    def build_from_paragraph(
        self,
        paragraph: ProcessedParagraph
    ) -> SemanticGraph:
        """Build semantic graph from a processed paragraph.

        Args:
            paragraph: Processed paragraph with sentences.

        Returns:
            SemanticGraph instance.
        """
        # Extract propositions from paragraph text
        propositions = self.prop_extractor.extract_from_text(paragraph.text)

        # Detect relationships
        relationships = self.rel_detector.detect_relationships(propositions)

        # Determine rhetorical intent
        intent = self._detect_intent(paragraph.text)

        # Map role string to enum
        role = ParagraphRole[paragraph.role]

        graph = SemanticGraph(
            nodes=propositions,
            edges=relationships,
            paragraph_idx=paragraph.index,
            role=role,
            intent=intent
        )

        logger.debug(
            f"Built graph for paragraph {paragraph.index}: "
            f"{len(propositions)} nodes, {len(relationships)} edges"
        )

        return graph

    def build_from_text(
        self,
        text: str,
        paragraph_idx: int = 0,
        role: str = "BODY"
    ) -> SemanticGraph:
        """Build semantic graph from raw text.

        Args:
            text: Paragraph text.
            paragraph_idx: Paragraph index in document.
            role: Paragraph role (INTRO, BODY, CONCLUSION).

        Returns:
            SemanticGraph instance.
        """
        # Extract propositions
        propositions = self.prop_extractor.extract_from_text(text)

        # Detect relationships
        relationships = self.rel_detector.detect_relationships(propositions)

        # Determine intent
        intent = self._detect_intent(text)

        # Map role
        role_enum = ParagraphRole[role] if role in ParagraphRole.__members__ else ParagraphRole.BODY

        return SemanticGraph(
            nodes=propositions,
            edges=relationships,
            paragraph_idx=paragraph_idx,
            role=role_enum,
            intent=intent
        )

    def _detect_intent(self, text: str) -> RhetoricalIntent:
        """Detect rhetorical intent of paragraph.

        Args:
            text: Paragraph text.

        Returns:
            RhetoricalIntent enum value.
        """
        text_lower = text.lower()

        # Question-heavy = INTERROGATIVE
        if text.count('?') >= 2:
            return RhetoricalIntent.INTERROGATIVE

        # Command/imperative markers
        imperative_starts = ['do', "don't", 'never', 'always', 'remember', 'consider', 'note']
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in imperative_starts:
            return RhetoricalIntent.IMPERATIVE

        # Definition markers
        definition_markers = ['is defined as', 'refers to', 'means that']
        if any(marker in text_lower for marker in definition_markers):
            return RhetoricalIntent.DEFINITION

        # Narrative markers
        narrative_markers = ['once upon', 'happened', 'occurred', 'story of']
        if any(marker in text_lower for marker in narrative_markers):
            return RhetoricalIntent.NARRATIVE

        # Default to ARGUMENT
        return RhetoricalIntent.ARGUMENT

    def validate_graph(self, graph: SemanticGraph) -> List[str]:
        """Validate a semantic graph for issues.

        Args:
            graph: Graph to validate.

        Returns:
            List of validation issues (empty if valid).
        """
        issues = []

        # Check for empty graph
        if not graph.nodes:
            issues.append("Graph has no proposition nodes")
            return issues

        # Check node IDs are unique
        node_ids = [n.id for n in graph.nodes]
        if len(node_ids) != len(set(node_ids)):
            issues.append("Duplicate node IDs found")

        # Check edges reference valid nodes
        valid_ids = set(node_ids)
        for edge in graph.edges:
            if edge.source_id not in valid_ids:
                issues.append(f"Edge references invalid source: {edge.source_id}")
            if edge.target_id not in valid_ids:
                issues.append(f"Edge references invalid target: {edge.target_id}")

        # Check for self-loops
        for edge in graph.edges:
            if edge.source_id == edge.target_id:
                issues.append(f"Self-loop detected: {edge.source_id}")

        return issues


class DocumentGraphBuilder:
    """Builds document-level graph from paragraphs.

    Combines paragraph-level semantic graphs into a document graph
    with thesis, intent, and cross-paragraph relationships.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        use_llm_relationships: bool = True
    ):
        """Initialize document graph builder.

        Args:
            llm_provider: Optional LLM provider.
            use_llm_relationships: Whether to use LLM for detection.
        """
        self.llm_provider = llm_provider
        self.graph_builder = SemanticGraphBuilder(
            llm_provider=llm_provider,
            use_llm_relationships=use_llm_relationships
        )

    def build_from_document(self, document: ProcessedDocument) -> DocumentGraph:
        """Build document graph from processed document.

        Args:
            document: Processed document with paragraphs.

        Returns:
            DocumentGraph instance.
        """
        # Build semantic graph for each paragraph
        paragraph_graphs = []
        for para in document.paragraphs:
            graph = self.graph_builder.build_from_paragraph(para)
            paragraph_graphs.append(graph)

        # Extract document-level context
        all_text = document.cleaned_text

        # Extract thesis (first significant sentence)
        thesis = self._extract_thesis(document)

        # Detect document intent
        intent = self._detect_document_intent(document)

        # Extract global keywords
        keywords = self._extract_global_keywords(paragraph_graphs)

        # Detect perspective
        perspective = self._detect_perspective(all_text)

        doc_graph = DocumentGraph(
            paragraphs=paragraph_graphs,
            thesis=thesis,
            intent=intent,
            keywords=keywords,
            perspective=perspective
        )

        logger.info(
            f"Built document graph: {len(paragraph_graphs)} paragraphs, "
            f"intent={intent}, perspective={perspective}"
        )

        return doc_graph

    def _extract_thesis(self, document: ProcessedDocument) -> str:
        """Extract the thesis statement from the document.

        Typically the first or second sentence of the introduction.

        Args:
            document: Processed document.

        Returns:
            Thesis string.
        """
        if not document.paragraphs:
            return ""

        # Get first paragraph (intro)
        intro = document.paragraphs[0]
        if intro.sentences:
            # Use second sentence if available (often more thesis-like)
            if len(intro.sentences) > 1:
                return intro.sentences[1]
            return intro.sentences[0]

        return ""

    def _detect_document_intent(self, document: ProcessedDocument) -> str:
        """Detect the overall intent of the document.

        Args:
            document: Processed document.

        Returns:
            Intent string (persuade, inform, narrate, explain).
        """
        all_text = document.cleaned_text.lower()

        # Check for persuasive markers
        persuasive = ['should', 'must', 'argue', 'believe', 'claim', 'prove']
        if sum(all_text.count(m) for m in persuasive) >= 3:
            return "persuade"

        # Check for narrative markers
        narrative = ['once', 'then', 'happened', 'story', 'told']
        if sum(all_text.count(m) for m in narrative) >= 3:
            return "narrate"

        # Check for explanatory markers
        explanatory = ['because', 'why', 'how', 'reason', 'explains']
        if sum(all_text.count(m) for m in explanatory) >= 3:
            return "explain"

        # Default to inform
        return "inform"

    def _extract_global_keywords(
        self,
        paragraph_graphs: List[SemanticGraph]
    ) -> List[str]:
        """Extract keywords that appear across paragraphs.

        Args:
            paragraph_graphs: List of paragraph semantic graphs.

        Returns:
            List of global keywords.
        """
        # Count keyword occurrences across paragraphs
        keyword_counts = {}
        for graph in paragraph_graphs:
            para_keywords = set()
            for node in graph.nodes:
                para_keywords.update(node.keywords)

            for kw in para_keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        # Filter to keywords appearing in multiple paragraphs
        global_keywords = [
            kw for kw, count in keyword_counts.items()
            if count >= 2
        ]

        # Sort by frequency
        global_keywords.sort(key=lambda kw: keyword_counts[kw], reverse=True)

        return global_keywords[:20]  # Top 20

    def _detect_perspective(self, text: str) -> str:
        """Detect narrative perspective.

        Args:
            text: Document text.

        Returns:
            Perspective string.
        """
        import re

        text_lower = text.lower()

        # Count pronouns
        first_singular = len(re.findall(r'\b(i|me|my|mine|myself)\b', text_lower))
        first_plural = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text_lower))
        third = len(re.findall(r'\b(he|she|it|they|him|her|them|his|hers|its|their)\b', text_lower))

        if first_singular > first_plural and first_singular > third:
            return "first_person_singular"
        elif first_plural > first_singular and first_plural > third:
            return "first_person_plural"
        else:
            return "third_person"
