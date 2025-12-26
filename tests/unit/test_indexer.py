"""Unit tests for ChromaDB indexer."""

import pytest
from unittest.mock import MagicMock, patch

from src.corpus.indexer import CorpusIndexer, StyleFragment, StyleGraph
from src.corpus.loader import CorpusLoader, Corpus
from src.corpus.preprocessor import ProcessedParagraph, ProcessedDocument
from src.config import ChromaDBConfig


class TestStyleFragment:
    """Test StyleFragment data class."""

    def test_create_fragment(self):
        """Test creating a style fragment."""
        fragment = StyleFragment(
            id="test123",
            text="Sample paragraph text.",
            author="test_author",
            role="BODY",
            sentence_count=2,
            avg_sentence_length=10.5,
            burstiness=0.3,
            document_id="doc123"
        )

        assert fragment.id == "test123"
        assert fragment.author == "test_author"
        assert fragment.role == "BODY"

    def test_to_metadata(self):
        """Test converting fragment to metadata."""
        fragment = StyleFragment(
            id="test123",
            text="Sample text.",
            author="test_author",
            role="INTRO",
            sentence_count=3,
            avg_sentence_length=15.0,
            burstiness=0.25,
            document_id="doc456"
        )

        metadata = fragment.to_metadata()

        assert metadata["author"] == "test_author"
        assert metadata["role"] == "INTRO"
        assert metadata["sentence_count"] == 3
        assert metadata["avg_sentence_length"] == 15.0
        assert metadata["burstiness"] == 0.25
        assert metadata["document_id"] == "doc456"


class TestStyleGraph:
    """Test StyleGraph data class."""

    def test_create_graph(self):
        """Test creating a style graph."""
        graph = StyleGraph(
            id="graph123",
            text="Original paragraph.",
            author="test_author",
            role="BODY",
            skeleton="[S1] [S2]",
            intent="ARGUMENT",
            signature="CAUSALITY",
            node_count=2,
            edge_types=["SEQUENCE", "CAUSALITY"],
            burstiness=0.4,
            document_id="doc789"
        )

        assert graph.id == "graph123"
        assert graph.skeleton == "[S1] [S2]"
        assert graph.intent == "ARGUMENT"
        assert graph.signature == "CAUSALITY"

    def test_to_metadata(self):
        """Test converting graph to metadata."""
        graph = StyleGraph(
            id="graph123",
            text="Original paragraph.",
            author="author",
            role="CONCLUSION",
            skeleton="[S1] [S2] [S3]",
            intent="NARRATIVE",
            signature="SEQUENCE",
            node_count=3,
            edge_types=["SEQUENCE"],
            burstiness=0.2,
            document_id="doc000"
        )

        metadata = graph.to_metadata()

        assert metadata["author"] == "author"
        assert metadata["role"] == "CONCLUSION"
        assert metadata["skeleton"] == "[S1] [S2] [S3]"
        assert metadata["intent"] == "NARRATIVE"
        assert metadata["signature"] == "SEQUENCE"
        assert metadata["node_count"] == 3
        # edge_types is JSON encoded
        assert "SEQUENCE" in metadata["edge_types"]


class TestCorpusIndexer:
    """Test CorpusIndexer functionality."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for testing without ChromaDB dependency."""
        config = ChromaDBConfig(persist_path=None)
        return CorpusIndexer(config=config)

    def test_generate_skeleton(self, indexer):
        """Test skeleton generation."""
        sentences = ["First sentence.", "Second sentence.", "Third one."]
        skeleton = indexer._generate_skeleton(sentences)

        assert "[S1]" in skeleton
        assert "[S2]" in skeleton
        assert "[S3]" in skeleton

    def test_detect_intent_interrogative(self, indexer):
        """Test interrogative intent detection."""
        text = "What is this? Why does it matter? How does it work?"
        intent = indexer._detect_intent(text)

        assert intent == "INTERROGATIVE"

    def test_detect_intent_imperative(self, indexer):
        """Test imperative intent detection."""
        text = "Consider the following implications."
        intent = indexer._detect_intent(text)

        assert intent == "IMPERATIVE"

    def test_detect_intent_definition(self, indexer):
        """Test definition intent detection."""
        text = "A variable is defined as a named storage location."
        intent = indexer._detect_intent(text)

        assert intent == "DEFINITION"

    def test_detect_intent_narrative(self, indexer):
        """Test narrative intent detection."""
        text = "Then the system processes the request. After that, it returns a response."
        intent = indexer._detect_intent(text)

        assert intent == "NARRATIVE"

    def test_detect_intent_argument(self, indexer):
        """Test default argument intent."""
        text = "We should adopt better practices for improved outcomes."
        intent = indexer._detect_intent(text)

        assert intent == "ARGUMENT"

    def test_detect_signature_contrast(self, indexer):
        """Test contrast signature detection."""
        text = "The old system was slow. However, the new one is much faster."
        signature = indexer._detect_signature(text)

        assert signature == "CONTRAST"

    def test_detect_signature_causality(self, indexer):
        """Test causality signature detection."""
        text = "The cache was full. Therefore, performance degraded."
        signature = indexer._detect_signature(text)

        assert signature == "CAUSALITY"

    def test_detect_signature_elaboration(self, indexer):
        """Test elaboration signature detection."""
        text = "Many options exist. For example, you could use Redis or Memcached."
        signature = indexer._detect_signature(text)

        assert signature == "ELABORATION"

    def test_detect_signature_sequence(self, indexer):
        """Test default sequence signature."""
        text = "Step one completes. Step two begins."
        signature = indexer._detect_signature(text)

        assert signature == "SEQUENCE"

    def test_compute_id(self, indexer):
        """Test ID computation."""
        id1 = indexer._compute_id("content", "author")
        id2 = indexer._compute_id("content", "author")
        id3 = indexer._compute_id("different", "author")

        # Same content should produce same ID
        assert id1 == id2
        # Different content should produce different ID
        assert id1 != id3
        # ID should be reasonable length
        assert len(id1) == 16

    def test_format_results_empty(self, indexer):
        """Test formatting empty results."""
        empty_results = {"ids": None}
        formatted = indexer._format_results(empty_results)

        assert formatted == []

    def test_format_results_with_data(self, indexer):
        """Test formatting results with data."""
        results = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"author": "a1"}, {"author": "a2"}]]
        }
        formatted = indexer._format_results(results)

        assert len(formatted) == 2
        assert formatted[0]["id"] == "id1"
        assert formatted[0]["text"] == "doc1"
        assert formatted[0]["distance"] == 0.1
        assert formatted[0]["metadata"]["author"] == "a1"


class TestIndexerFragmentCreation:
    """Test fragment and graph creation logic."""

    @pytest.fixture
    def sample_corpus(self):
        """Create a sample corpus for testing."""
        loader = CorpusLoader()

        text = """First paragraph introduction.

        Second paragraph with body content. It has multiple sentences. Including this one.

        Third paragraph conclusion."""

        doc = loader.load_text(text, author_name="test_author")
        corpus = Corpus(documents=[doc], author="test_author")

        return corpus

    def test_create_fragment(self, sample_corpus):
        """Test fragment creation from paragraph."""
        indexer = CorpusIndexer()
        doc = sample_corpus.documents[0]
        para = doc.processed.paragraphs[0]

        fragment = indexer._create_fragment(para, doc)

        assert fragment.id is not None
        assert fragment.author == "test_author"
        assert fragment.role == "INTRO"
        assert fragment.sentence_count == 1

    def test_create_graph(self, sample_corpus):
        """Test graph creation from paragraph."""
        indexer = CorpusIndexer()
        doc = sample_corpus.documents[0]
        para = doc.processed.paragraphs[1]  # Body paragraph

        graph = indexer._create_graph(para, doc)

        assert graph.id is not None
        assert graph.author == "test_author"
        assert graph.role == "BODY"
        assert "[S" in graph.skeleton  # Has sentence placeholders
        assert graph.intent in ["ARGUMENT", "NARRATIVE", "DEFINITION", "INTERROGATIVE", "IMPERATIVE"]
        assert graph.signature in ["SEQUENCE", "CONTRAST", "CAUSALITY", "ELABORATION"]
