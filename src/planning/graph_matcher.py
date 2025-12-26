"""Style graph matching using ChromaDB."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ..models.graph import SemanticGraph, RelationshipType
from ..models.style import StyleProfile
from ..corpus.indexer import CorpusIndexer
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MatchedStyleGraph:
    """A matched style graph from ChromaDB with relevance info."""
    id: str
    text: str
    skeleton: str
    intent: str
    signature: str
    author: str
    role: str
    burstiness: float
    node_count: int
    edge_types: List[str]
    similarity_score: float
    edge_overlap_score: float

    @property
    def combined_score(self) -> float:
        """Combined relevance score."""
        return 0.6 * self.similarity_score + 0.4 * self.edge_overlap_score


class GraphMatcher:
    """Matches semantic graphs to style templates in ChromaDB.

    Uses a multi-stage strategy:
    1. Metadata filtering (intent, role, complexity)
    2. Semantic similarity search
    3. Re-ranking by edge type overlap
    """

    def __init__(
        self,
        indexer: Optional[CorpusIndexer] = None,
        fallback_enabled: bool = True
    ):
        """Initialize graph matcher.

        Args:
            indexer: ChromaDB indexer instance.
            fallback_enabled: Whether to use fallback when no matches found.
        """
        self.indexer = indexer
        self.fallback_enabled = fallback_enabled

    def find_matches(
        self,
        semantic_graph: SemanticGraph,
        style_profile: StyleProfile,
        n_results: int = 5
    ) -> List[MatchedStyleGraph]:
        """Find matching style graphs for a semantic graph.

        Args:
            semantic_graph: Source semantic graph to match.
            style_profile: Target author style profile.
            n_results: Number of results to return.

        Returns:
            List of matched style graphs, sorted by relevance.
        """
        if not self.indexer:
            logger.warning("No indexer configured, returning empty matches")
            return []

        author_name = style_profile.get_author_name()

        # Create query text from semantic graph
        query_text = self._create_query_text(semantic_graph)

        # Query ChromaDB with filters
        try:
            results = self.indexer.query_graphs(
                query_text=query_text,
                author=author_name,
                intent=semantic_graph.intent.value,
                n_results=n_results * 2  # Get more for re-ranking
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        if not results:
            logger.debug(f"No matches found for {author_name}")
            return []

        # Convert to MatchedStyleGraph objects
        matches = []
        for result in results:
            match = self._result_to_match(result, semantic_graph)
            if match:
                matches.append(match)

        # Sort by combined score
        matches.sort(key=lambda m: m.combined_score, reverse=True)

        logger.debug(f"Found {len(matches)} matches for semantic graph")

        return matches[:n_results]

    def find_best_match(
        self,
        semantic_graph: SemanticGraph,
        style_profile: StyleProfile
    ) -> Optional[MatchedStyleGraph]:
        """Find the single best matching style graph.

        Args:
            semantic_graph: Source semantic graph.
            style_profile: Target style profile.

        Returns:
            Best matching style graph or None.
        """
        matches = self.find_matches(semantic_graph, style_profile, n_results=1)
        return matches[0] if matches else None

    def _create_query_text(self, graph: SemanticGraph) -> str:
        """Create query text from semantic graph.

        Concatenates proposition texts with relationship context.

        Args:
            graph: Semantic graph to convert.

        Returns:
            Query text string.
        """
        parts = []

        # Add proposition texts
        for node in graph.nodes:
            parts.append(node.text)

        # Add relationship context
        for edge in graph.edges:
            source = self._find_node(graph, edge.source_id)
            target = self._find_node(graph, edge.target_id)
            if source and target:
                parts.append(
                    f"{source.text} [{edge.relationship.value}] {target.text}"
                )

        return " ".join(parts)

    def _find_node(self, graph: SemanticGraph, node_id: str):
        """Find a node in the graph by ID."""
        for node in graph.nodes:
            if node.id == node_id:
                return node
        return None

    def _result_to_match(
        self,
        result: Dict[str, Any],
        semantic_graph: SemanticGraph
    ) -> Optional[MatchedStyleGraph]:
        """Convert ChromaDB result to MatchedStyleGraph.

        Args:
            result: ChromaDB query result.
            semantic_graph: Source graph for edge overlap calculation.

        Returns:
            MatchedStyleGraph or None if invalid.
        """
        metadata = result.get("metadata", {})
        if not metadata:
            return None

        # Parse edge_types from JSON string
        import json
        edge_types_str = metadata.get("edge_types", "[]")
        try:
            edge_types = json.loads(edge_types_str)
        except json.JSONDecodeError:
            edge_types = []

        # Calculate edge overlap score
        edge_overlap = self._calculate_edge_overlap(
            source_edges=[e.relationship.value for e in semantic_graph.edges],
            target_edges=edge_types
        )

        # Similarity score from ChromaDB distance
        distance = result.get("distance", 1.0)
        similarity = 1.0 - min(distance, 1.0)  # Convert distance to similarity

        return MatchedStyleGraph(
            id=result.get("id", ""),
            text=result.get("text", ""),
            skeleton=metadata.get("skeleton", ""),
            intent=metadata.get("intent", ""),
            signature=metadata.get("signature", ""),
            author=metadata.get("author", ""),
            role=metadata.get("role", ""),
            burstiness=metadata.get("burstiness", 0.0),
            node_count=metadata.get("node_count", 0),
            edge_types=edge_types,
            similarity_score=similarity,
            edge_overlap_score=edge_overlap
        )

    def _calculate_edge_overlap(
        self,
        source_edges: List[str],
        target_edges: List[str]
    ) -> float:
        """Calculate overlap between edge types.

        Args:
            source_edges: Edge types from source graph.
            target_edges: Edge types from target graph.

        Returns:
            Overlap score between 0 and 1.
        """
        if not source_edges and not target_edges:
            return 1.0  # Both empty = perfect match

        if not source_edges or not target_edges:
            return 0.5  # One empty = partial match

        source_set = set(source_edges)
        target_set = set(target_edges)

        intersection = source_set & target_set
        union = source_set | target_set

        if not union:
            return 1.0

        return len(intersection) / len(union)  # Jaccard similarity


class FallbackMatcher:
    """Fallback matcher when no ChromaDB matches found.

    Provides default sentence structure based on author statistics.
    """

    def __init__(self, style_profile: StyleProfile):
        """Initialize fallback matcher.

        Args:
            style_profile: Target style profile.
        """
        self.style_profile = style_profile

    def create_default_structure(
        self,
        semantic_graph: SemanticGraph
    ) -> Dict[str, Any]:
        """Create default sentence structure.

        Args:
            semantic_graph: Source semantic graph.

        Returns:
            Default structure info.
        """
        num_props = len(semantic_graph.nodes)
        avg_length = self.style_profile.get_effective_avg_sentence_length()
        burstiness = self.style_profile.get_effective_burstiness()

        # Estimate number of sentences
        props_per_sentence = max(1, int(avg_length / 8))  # ~8 words per proposition
        num_sentences = max(1, (num_props + props_per_sentence - 1) // props_per_sentence)

        # Generate sentence length pattern based on burstiness
        lengths = self._generate_length_pattern(num_sentences, avg_length, burstiness)

        return {
            "num_sentences": num_sentences,
            "target_lengths": lengths,
            "skeleton": None,  # No skeleton in fallback
            "is_fallback": True
        }

    def _generate_length_pattern(
        self,
        num_sentences: int,
        avg_length: float,
        burstiness: float
    ) -> List[int]:
        """Generate sentence length pattern.

        Args:
            num_sentences: Number of sentences.
            avg_length: Average target length.
            burstiness: Target burstiness (variation).

        Returns:
            List of target lengths.
        """
        import random

        if num_sentences == 1:
            return [int(avg_length)]

        # Calculate standard deviation from burstiness
        std_dev = avg_length * burstiness

        lengths = []
        for i in range(num_sentences):
            # Generate varied lengths
            if burstiness < 0.1:
                # Low burstiness = uniform
                length = int(avg_length)
            else:
                # Higher burstiness = more variation
                length = int(random.gauss(avg_length, std_dev))
                length = max(5, min(length, int(avg_length * 2)))  # Clamp
            lengths.append(length)

        return lengths
