"""ChromaDB indexer for style graphs and fragments."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..config import ChromaDBConfig
from ..utils.logging import get_logger
from .loader import Corpus, CorpusDocument
from .preprocessor import ProcessedParagraph
from .analyzer import StatisticalAnalyzer, FeatureVector

logger = get_logger(__name__)


@dataclass
class StyleFragment:
    """A fragment of styled text with metadata."""
    id: str
    text: str
    author: str
    role: str  # INTRO, BODY, CONCLUSION
    sentence_count: int
    avg_sentence_length: float
    burstiness: float
    document_id: str

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata format."""
        return {
            "author": self.author,
            "role": self.role,
            "sentence_count": self.sentence_count,
            "avg_sentence_length": self.avg_sentence_length,
            "burstiness": self.burstiness,
            "document_id": self.document_id,
        }


@dataclass
class StyleGraph:
    """A graph structure representing paragraph topology."""
    id: str
    text: str  # Original paragraph text
    author: str
    role: str
    skeleton: str  # Syntactic template with placeholders
    intent: str  # Rhetorical intent (ARGUMENT, NARRATIVE, etc.)
    signature: str  # Logical signature (CONTRAST, CAUSALITY, etc.)
    node_count: int
    edge_types: List[str]
    burstiness: float
    document_id: str

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata format."""
        return {
            "author": self.author,
            "role": self.role,
            "skeleton": self.skeleton,
            "intent": self.intent,
            "signature": self.signature,
            "node_count": self.node_count,
            "edge_types": json.dumps(self.edge_types),
            "burstiness": self.burstiness,
            "document_id": self.document_id,
        }


class CorpusIndexer:
    """Indexes corpus into ChromaDB collections.

    Creates two collections:
    - style_fragments: Paragraph-level text with style metadata
    - style_graphs: Graph topologies with skeletons for template matching
    """

    def __init__(
        self,
        config: Optional[ChromaDBConfig] = None,
        analyzer: Optional[StatisticalAnalyzer] = None
    ):
        """Initialize indexer.

        Args:
            config: ChromaDB configuration.
            analyzer: Statistical analyzer instance.
        """
        self.config = config or ChromaDBConfig()
        self.analyzer = analyzer or StatisticalAnalyzer()
        self._client = None
        self._fragments_collection = None
        self._graphs_collection = None

    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )

            if self.config.persist_path:
                self._client = chromadb.PersistentClient(
                    path=self.config.persist_path,
                    settings=settings
                )
                logger.info(f"ChromaDB initialized at: {self.config.persist_path}")
            else:
                self._client = chromadb.Client(settings=settings)
                logger.info("ChromaDB initialized in-memory")

        return self._client

    @property
    def fragments_collection(self):
        """Get or create the style_fragments collection."""
        if self._fragments_collection is None:
            self._fragments_collection = self.client.get_or_create_collection(
                name="style_fragments",
                metadata={"hnsw:space": "cosine"}
            )
        return self._fragments_collection

    @property
    def graphs_collection(self):
        """Get or create the style_graphs collection."""
        if self._graphs_collection is None:
            self._graphs_collection = self.client.get_or_create_collection(
                name="style_graphs",
                metadata={"hnsw:space": "cosine"}
            )
        return self._graphs_collection

    def index_corpus(self, corpus: Corpus) -> Dict[str, int]:
        """Index an entire corpus.

        Args:
            corpus: Corpus to index.

        Returns:
            Dictionary with count of indexed items.
        """
        fragments_indexed = 0
        graphs_indexed = 0

        for doc in corpus.documents:
            doc_counts = self.index_document(doc)
            fragments_indexed += doc_counts["fragments"]
            graphs_indexed += doc_counts["graphs"]

        logger.info(
            f"Indexed corpus: {fragments_indexed} fragments, {graphs_indexed} graphs"
        )

        return {
            "fragments": fragments_indexed,
            "graphs": graphs_indexed
        }

    def index_document(self, doc: CorpusDocument) -> Dict[str, int]:
        """Index a single document.

        Args:
            doc: Document to index.

        Returns:
            Dictionary with count of indexed items.
        """
        fragments_indexed = 0
        graphs_indexed = 0

        for para in doc.processed.paragraphs:
            # Create and index fragment
            fragment = self._create_fragment(para, doc)
            if self._index_fragment(fragment):
                fragments_indexed += 1

            # Create and index graph (simplified for now)
            graph = self._create_graph(para, doc)
            if self._index_graph(graph):
                graphs_indexed += 1

        return {
            "fragments": fragments_indexed,
            "graphs": graphs_indexed
        }

    def _create_fragment(
        self,
        para: ProcessedParagraph,
        doc: CorpusDocument
    ) -> StyleFragment:
        """Create a StyleFragment from a paragraph.

        Args:
            para: Processed paragraph.
            doc: Parent document.

        Returns:
            StyleFragment instance.
        """
        features = self.analyzer.analyze_paragraph(para)

        fragment_id = self._compute_id(para.text, doc.author)

        return StyleFragment(
            id=fragment_id,
            text=para.text,
            author=doc.author,
            role=para.role,
            sentence_count=len(para.sentences),
            avg_sentence_length=features.avg_sentence_length,
            burstiness=features.burstiness,
            document_id=doc.id
        )

    def _create_graph(
        self,
        para: ProcessedParagraph,
        doc: CorpusDocument
    ) -> StyleGraph:
        """Create a StyleGraph from a paragraph.

        For now, creates a simplified skeleton. Full graph extraction
        will be implemented with LLM in later phases.

        Args:
            para: Processed paragraph.
            doc: Parent document.

        Returns:
            StyleGraph instance.
        """
        features = self.analyzer.analyze_paragraph(para)

        # Generate skeleton (simplified - replace nouns/verbs with placeholders)
        skeleton = self._generate_skeleton(para.sentences)

        # Detect intent (simplified heuristic)
        intent = self._detect_intent(para.text)

        # Detect signature (simplified heuristic)
        signature = self._detect_signature(para.text)

        graph_id = self._compute_id(para.text + "_graph", doc.author)

        return StyleGraph(
            id=graph_id,
            text=para.text,
            author=doc.author,
            role=para.role,
            skeleton=skeleton,
            intent=intent,
            signature=signature,
            node_count=len(para.sentences),
            edge_types=["SEQUENCE"],  # Default to sequence
            burstiness=features.burstiness,
            document_id=doc.id
        )

    def _generate_skeleton(self, sentences: List[str]) -> str:
        """Generate a syntactic skeleton from sentences.

        Replaces content words with [S1], [S2], etc. placeholders.

        Args:
            sentences: List of sentences.

        Returns:
            Skeleton template string.
        """
        parts = []
        for i, sent in enumerate(sentences):
            # For now, just create sentence placeholders
            # Full implementation will use POS tagging to replace content words
            parts.append(f"[S{i+1}]")

        return " ".join(parts)

    def _detect_intent(self, text: str) -> str:
        """Detect rhetorical intent using heuristics.

        Args:
            text: Paragraph text.

        Returns:
            Intent classification.
        """
        text_lower = text.lower()

        # Question-heavy = INTERROGATIVE
        if text.count('?') >= 2:
            return "INTERROGATIVE"

        # Command/imperative markers
        imperative_starts = ['do', 'don\'t', 'never', 'always', 'remember', 'consider', 'note']
        if any(text_lower.startswith(word) for word in imperative_starts):
            return "IMPERATIVE"

        # Definition markers
        definition_markers = ['is defined as', 'refers to', 'means that', 'is a', 'are a']
        if any(marker in text_lower for marker in definition_markers):
            return "DEFINITION"

        # Narrative markers (past tense, temporal sequences)
        narrative_markers = ['then', 'after', 'before', 'once', 'when', 'while']
        if any(marker in text_lower for marker in narrative_markers):
            return "NARRATIVE"

        # Default to ARGUMENT (most common in analytical writing)
        return "ARGUMENT"

    def _detect_signature(self, text: str) -> str:
        """Detect logical signature using heuristics.

        Args:
            text: Paragraph text.

        Returns:
            Signature classification.
        """
        text_lower = text.lower()

        # Contrast markers
        contrast_markers = ['however', 'but', 'although', 'despite', 'yet', 'on the other hand', 'contrary']
        if any(marker in text_lower for marker in contrast_markers):
            return "CONTRAST"

        # Causality markers
        causality_markers = ['because', 'therefore', 'thus', 'hence', 'consequently', 'as a result', 'due to']
        if any(marker in text_lower for marker in causality_markers):
            return "CAUSALITY"

        # Elaboration markers
        elaboration_markers = ['for example', 'for instance', 'such as', 'namely', 'specifically']
        if any(marker in text_lower for marker in elaboration_markers):
            return "ELABORATION"

        # Default to SEQUENCE
        return "SEQUENCE"

    def _index_fragment(self, fragment: StyleFragment) -> bool:
        """Index a fragment into ChromaDB.

        Args:
            fragment: Fragment to index.

        Returns:
            True if indexed successfully.
        """
        try:
            # Check if already exists (deduplication)
            existing = self.fragments_collection.get(ids=[fragment.id])
            if existing and existing['ids']:
                logger.debug(f"Fragment already exists: {fragment.id}")
                return False

            self.fragments_collection.add(
                ids=[fragment.id],
                documents=[fragment.text],
                metadatas=[fragment.to_metadata()]
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index fragment {fragment.id}: {e}")
            return False

    def _index_graph(self, graph: StyleGraph) -> bool:
        """Index a graph into ChromaDB.

        Args:
            graph: Graph to index.

        Returns:
            True if indexed successfully.
        """
        try:
            # Check if already exists (deduplication)
            existing = self.graphs_collection.get(ids=[graph.id])
            if existing and existing['ids']:
                logger.debug(f"Graph already exists: {graph.id}")
                return False

            self.graphs_collection.add(
                ids=[graph.id],
                documents=[graph.text],
                metadatas=[graph.to_metadata()]
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index graph {graph.id}: {e}")
            return False

    def query_fragments(
        self,
        query_text: str,
        author: Optional[str] = None,
        role: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """Query fragments collection.

        Args:
            query_text: Text to search for.
            author: Optional author filter.
            role: Optional role filter.
            n_results: Number of results to return.

        Returns:
            List of matching fragments with metadata.
        """
        where_filter = self._build_where_filter(author=author, role=role)

        results = self.fragments_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        return self._format_results(results)

    def query_graphs(
        self,
        query_text: str,
        author: Optional[str] = None,
        intent: Optional[str] = None,
        signature: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """Query graphs collection.

        Args:
            query_text: Text to search for.
            author: Optional author filter.
            intent: Optional intent filter.
            signature: Optional signature filter.
            n_results: Number of results to return.

        Returns:
            List of matching graphs with metadata.
        """
        where_filter = self._build_where_filter(
            author=author, intent=intent, signature=signature
        )

        results = self.graphs_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        return self._format_results(results)

    def _build_where_filter(self, **kwargs) -> Optional[Dict]:
        """Build ChromaDB where filter from keyword arguments.

        Handles single conditions directly and multiple conditions with $and.

        Args:
            **kwargs: Field-value pairs to filter on.

        Returns:
            ChromaDB where filter or None if no conditions.
        """
        conditions = []
        for key, value in kwargs.items():
            if value is not None:
                conditions.append({key: value})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results into a cleaner structure.

        Args:
            results: Raw ChromaDB results.

        Returns:
            List of result dictionaries.
        """
        formatted = []
        if not results or not results.get('ids'):
            return formatted

        for i, id_ in enumerate(results['ids'][0]):
            result = {
                "id": id_,
                "text": results['documents'][0][i] if results.get('documents') else None,
                "distance": results['distances'][0][i] if results.get('distances') else None,
            }
            if results.get('metadatas'):
                result["metadata"] = results['metadatas'][0][i]
            formatted.append(result)

        return formatted

    def _compute_id(self, content: str, author: str) -> str:
        """Compute unique ID for content.

        Args:
            content: Content to hash.
            author: Author name.

        Returns:
            Unique ID string.
        """
        combined = f"{author}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get_stats(self) -> Dict[str, int]:
        """Get collection statistics.

        Returns:
            Dictionary with collection counts.
        """
        return {
            "fragments": self.fragments_collection.count(),
            "graphs": self.graphs_collection.count()
        }

    def clear(self) -> None:
        """Clear all collections."""
        self.client.delete_collection("style_fragments")
        self.client.delete_collection("style_graphs")
        self._fragments_collection = None
        self._graphs_collection = None
        logger.info("Cleared all collections")
