"""Style retriever for RAG-based style context.

Implements two-channel retrieval:
1. Semantic: Find chunks about similar topics
2. Structural: Filter by style metrics similarity
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..utils.logging import get_logger
from .corpus_indexer import CorpusIndexer, get_indexer
from .style_analyzer import StyleAnalyzer, StyleMetrics, analyze_style

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved style example."""

    text: str
    author: str
    semantic_score: float
    structural_score: float
    combined_score: float

    @property
    def is_good_match(self) -> bool:
        """Check if this is a good stylistic match."""
        return self.combined_score > 0.5


class StyleRetriever:
    """Retrieves stylistically relevant chunks for few-shot prompting."""

    def __init__(
        self,
        indexer: Optional[CorpusIndexer] = None,
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4,
    ):
        """Initialize the retriever.

        Args:
            indexer: CorpusIndexer instance. If None, uses default.
            semantic_weight: Weight for semantic similarity (0-1).
            structural_weight: Weight for structural similarity (0-1).
        """
        self._indexer = indexer
        self._analyzer = StyleAnalyzer()
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self._embedding_model = None

    @property
    def indexer(self) -> CorpusIndexer:
        """Get the corpus indexer."""
        if self._indexer is None:
            self._indexer = get_indexer()
        return self._indexer

    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            from .corpus_indexer import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def retrieve(
        self,
        input_text: str,
        author: str,
        k: int = 3,
        candidates: int = 20,
    ) -> List[RetrievedChunk]:
        """Retrieve style examples matching input text.

        Uses two-channel retrieval:
        1. Semantic search for topic similarity
        2. Structural filtering for style similarity

        Args:
            input_text: The text being styled (for context matching).
            author: Author whose corpus to search.
            k: Number of examples to return.
            candidates: Number of semantic candidates to consider.

        Returns:
            List of RetrievedChunk with style examples.
        """
        # Check if author is indexed
        chunk_count = self.indexer.get_chunk_count(author)
        if chunk_count == 0:
            logger.warning(f"No indexed chunks for author: {author}")
            return []

        logger.info(f"Retrieving style examples for '{author}' from {chunk_count} chunks")

        # Get input style metrics
        input_metrics = self._analyzer.analyze(input_text)

        # Generate input embedding
        input_embedding = self.embedding_model.encode([input_text])[0].tolist()

        # Semantic search - get more candidates than needed
        results = self.indexer.collection.query(
            query_embeddings=[input_embedding],
            n_results=min(candidates, chunk_count),
            where={"author": author},
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            logger.warning(f"No results from semantic search for {author}")
            return []

        # Build scored chunks
        chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert distance to similarity (ChromaDB returns L2 distance)
            # Lower distance = higher similarity
            semantic_score = 1.0 / (1.0 + dist)

            # Get structural similarity
            chunk_metrics = StyleMetrics.from_dict(meta)
            structural_distance = input_metrics.structural_distance(chunk_metrics)
            structural_score = 1.0 / (1.0 + structural_distance)

            # Combined score
            combined = (
                self.semantic_weight * semantic_score +
                self.structural_weight * structural_score
            )

            chunks.append(RetrievedChunk(
                text=doc,
                author=author,
                semantic_score=semantic_score,
                structural_score=structural_score,
                combined_score=combined,
            ))

        # Sort by combined score and return top k
        chunks.sort(key=lambda c: c.combined_score, reverse=True)
        top_chunks = chunks[:k]

        logger.info(
            f"Retrieved {len(top_chunks)} examples. "
            f"Best score: {top_chunks[0].combined_score:.3f}" if top_chunks else ""
        )

        return top_chunks

    def retrieve_by_structure(
        self,
        target_metrics: StyleMetrics,
        author: str,
        k: int = 3,
    ) -> List[RetrievedChunk]:
        """Retrieve chunks matching target structural metrics.

        Useful when you want specific sentence patterns regardless of topic.

        Args:
            target_metrics: Target style metrics to match.
            author: Author to search.
            k: Number of examples.

        Returns:
            List of structurally similar chunks.
        """
        # Get all chunks for author
        results = self.indexer.collection.get(
            where={"author": author},
            include=["documents", "metadatas"],
        )

        if not results["documents"]:
            return []

        # Score by structural similarity
        chunks = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunk_metrics = StyleMetrics.from_dict(meta)
            distance = target_metrics.structural_distance(chunk_metrics)
            score = 1.0 / (1.0 + distance)

            chunks.append(RetrievedChunk(
                text=doc,
                author=author,
                semantic_score=0.0,  # Not used
                structural_score=score,
                combined_score=score,
            ))

        chunks.sort(key=lambda c: c.combined_score, reverse=True)
        return chunks[:k]

    def retrieve_diverse(
        self,
        input_text: str,
        author: str,
        k: int = 3,
    ) -> List[RetrievedChunk]:
        """Retrieve diverse style examples (variety of sentence structures).

        Ensures examples cover different stylistic patterns from the author.

        Args:
            input_text: Input text for context.
            author: Author to search.
            k: Number of examples.

        Returns:
            List of diverse style examples.
        """
        # Get more candidates
        candidates = self.retrieve(input_text, author, k=k * 3, candidates=50)

        if len(candidates) <= k:
            return candidates

        # Select diverse examples by maximizing structural variety
        selected = [candidates[0]]  # Always include best match
        remaining = candidates[1:]

        while len(selected) < k and remaining:
            # Find candidate most different from selected
            best_diff = -1
            best_idx = 0

            for i, cand in enumerate(remaining):
                cand_metrics = analyze_style(cand.text)
                # Min distance to any selected
                min_dist = float("inf")
                for sel in selected:
                    sel_metrics = analyze_style(sel.text)
                    dist = cand_metrics.structural_distance(sel_metrics)
                    min_dist = min(min_dist, dist)

                # We want high diversity (high min distance)
                if min_dist > best_diff:
                    best_diff = min_dist
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected


# Module singleton
_retriever = None


def get_retriever(
    indexer: Optional[CorpusIndexer] = None,
    semantic_weight: float = 0.6,
    structural_weight: float = 0.4,
) -> StyleRetriever:
    """Get a style retriever.

    Args:
        indexer: Optional custom indexer.
        semantic_weight: Semantic similarity weight.
        structural_weight: Structural similarity weight.

    Returns:
        StyleRetriever instance.
    """
    global _retriever

    if indexer is not None:
        return StyleRetriever(indexer, semantic_weight, structural_weight)

    if _retriever is None:
        _retriever = StyleRetriever(
            semantic_weight=semantic_weight,
            structural_weight=structural_weight,
        )

    return _retriever
