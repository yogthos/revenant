"""Tests for the RAG (Retrieval-Augmented Generation) pipeline.

Tests cover:
- StyleAnalyzer: spaCy-based style metric extraction
- CorpusIndexer: ChromaDB indexing with embeddings
- StyleRetriever: Two-channel retrieval (semantic + structural)
- SessionContext: Session context management and prompt formatting
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil


# =============================================================================
# Tests for StyleAnalyzer
# =============================================================================

class TestStyleMetrics:
    """Tests for StyleMetrics dataclass."""

    def test_style_metrics_to_dict(self):
        """Test conversion to dictionary for ChromaDB metadata."""
        from src.rag.style_analyzer import StyleMetrics

        metrics = StyleMetrics(
            avg_sentence_length=15.5,
            sentence_length_std=8.2,
            dependency_depth=3.5,
            adjective_ratio=0.12,
            verb_ratio=0.18,
            punctuation_density=2.5,
            avg_word_length=5.2,
        )

        d = metrics.to_dict()

        assert d["avg_sentence_length"] == 15.5
        assert d["sentence_length_std"] == 8.2
        assert d["dependency_depth"] == 3.5
        assert d["adjective_ratio"] == 0.12
        assert d["verb_ratio"] == 0.18
        assert d["punctuation_density"] == 2.5
        assert d["avg_word_length"] == 5.2

    def test_style_metrics_from_dict(self):
        """Test creation from dictionary."""
        from src.rag.style_analyzer import StyleMetrics

        d = {
            "avg_sentence_length": 20.0,
            "sentence_length_std": 10.0,
            "dependency_depth": 4.0,
            "adjective_ratio": 0.15,
            "verb_ratio": 0.20,
            "punctuation_density": 3.0,
            "avg_word_length": 6.0,
        }

        metrics = StyleMetrics.from_dict(d)

        assert metrics.avg_sentence_length == 20.0
        assert metrics.adjective_ratio == 0.15

    def test_style_metrics_from_dict_with_missing_keys(self):
        """Test that missing keys default to 0.0."""
        from src.rag.style_analyzer import StyleMetrics

        d = {"avg_sentence_length": 15.0}  # Most keys missing

        metrics = StyleMetrics.from_dict(d)

        assert metrics.avg_sentence_length == 15.0
        assert metrics.dependency_depth == 0.0
        assert metrics.adjective_ratio == 0.0

    def test_structural_distance_identical(self):
        """Test that identical metrics have zero distance."""
        from src.rag.style_analyzer import StyleMetrics

        m1 = StyleMetrics(15.0, 5.0, 3.0, 0.1, 0.2, 2.0, 5.0)
        m2 = StyleMetrics(15.0, 5.0, 3.0, 0.1, 0.2, 2.0, 5.0)

        distance = m1.structural_distance(m2)

        assert distance == 0.0

    def test_structural_distance_different(self):
        """Test that different metrics have positive distance."""
        from src.rag.style_analyzer import StyleMetrics

        m1 = StyleMetrics(15.0, 5.0, 3.0, 0.1, 0.2, 2.0, 5.0)
        m2 = StyleMetrics(30.0, 10.0, 6.0, 0.2, 0.3, 4.0, 7.0)

        distance = m1.structural_distance(m2)

        assert distance > 0
        assert distance <= 1.0  # Normalized

    def test_structural_distance_symmetric(self):
        """Test that distance is symmetric."""
        from src.rag.style_analyzer import StyleMetrics

        m1 = StyleMetrics(15.0, 5.0, 3.0, 0.1, 0.2, 2.0, 5.0)
        m2 = StyleMetrics(25.0, 8.0, 4.0, 0.15, 0.25, 3.0, 6.0)

        d1 = m1.structural_distance(m2)
        d2 = m2.structural_distance(m1)

        assert abs(d1 - d2) < 0.001


class TestStyleAnalyzer:
    """Tests for StyleAnalyzer class."""

    def test_analyze_empty_text(self):
        """Test that empty text returns zero metrics."""
        from src.rag.style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer()
        metrics = analyzer.analyze("")

        assert metrics.avg_sentence_length == 0.0
        assert metrics.dependency_depth == 0.0

    def test_analyze_whitespace_only(self):
        """Test that whitespace-only text returns zero metrics."""
        from src.rag.style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer()
        metrics = analyzer.analyze("   \n\t  ")

        assert metrics.avg_sentence_length == 0.0

    def test_analyze_single_sentence(self):
        """Test analysis of a single sentence."""
        from src.rag.style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer()
        text = "The quick brown fox jumps over the lazy dog."
        metrics = analyzer.analyze(text)

        assert metrics.avg_sentence_length == 9  # 9 words
        assert metrics.sentence_length_std == 0.0  # Only one sentence
        assert metrics.dependency_depth > 0
        assert metrics.avg_word_length > 0

    def test_analyze_multiple_sentences(self):
        """Test analysis with multiple sentences."""
        from src.rag.style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer()
        text = "Short sentence. This is a much longer sentence with more words in it."
        metrics = analyzer.analyze(text)

        # Should have non-zero std dev with different sentence lengths
        assert metrics.sentence_length_std > 0
        # Average should be between 2 and 12
        assert 2 < metrics.avg_sentence_length < 12

    def test_analyze_punctuation_density(self):
        """Test punctuation density calculation."""
        from src.rag.style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer()

        # Text with lots of punctuation
        text_heavy = "Wait—no; stop, please—I can't; it's impossible, really—"
        metrics_heavy = analyzer.analyze(text_heavy)

        # Text with minimal punctuation
        text_light = "The sun rises in the east and sets in the west every day"
        metrics_light = analyzer.analyze(text_light)

        assert metrics_heavy.punctuation_density > metrics_light.punctuation_density

    def test_analyze_batch(self):
        """Test batch analysis of multiple texts."""
        from src.rag.style_analyzer import StyleAnalyzer

        analyzer = StyleAnalyzer()
        texts = [
            "First text here.",
            "Second text with more words.",
            "Third text.",
        ]

        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert all(r.avg_sentence_length > 0 for r in results)

    def test_get_style_analyzer_singleton(self):
        """Test that get_style_analyzer returns singleton."""
        from src.rag.style_analyzer import get_style_analyzer

        a1 = get_style_analyzer()
        a2 = get_style_analyzer()

        assert a1 is a2

    def test_analyze_style_convenience_function(self):
        """Test the module-level convenience function."""
        from src.rag.style_analyzer import analyze_style

        metrics = analyze_style("A simple test sentence.")

        assert metrics.avg_sentence_length > 0


# =============================================================================
# Tests for CorpusIndexer
# =============================================================================

class TestCorpusIndexer:
    """Tests for CorpusIndexer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        dirpath = tempfile.mkdtemp()
        yield dirpath
        shutil.rmtree(dirpath)

    @pytest.fixture
    def sample_corpus(self, temp_dir):
        """Create a sample corpus file."""
        corpus_path = Path(temp_dir) / "corpus.txt"
        corpus_text = """
The cosmos is all that is or was or ever will be. Our feeblest contemplations of the Cosmos stir us. There is a tingling in the spine, a catch in the voice, a faint sensation, as if a distant memory, of falling from a height. We know we are approaching the greatest of mysteries.

The size and age of the Cosmos are beyond ordinary human understanding. Lost somewhere between immensity and eternity is our tiny planetary home. In a cosmic perspective, most human concerns seem insignificant, even petty.

Science is not only compatible with spirituality; it is a profound source of spirituality. When we recognize our place in an immensity of light-years and in the passage of ages, when we grasp the intricacy, beauty, and subtlety of life, then that soaring feeling, that sense of elation and humility combined, is surely spiritual.
        """.strip()
        corpus_path.write_text(corpus_text)
        return str(corpus_path)

    def test_split_into_chunks_basic(self, temp_dir):
        """Test basic chunking of text."""
        from src.rag.corpus_indexer import CorpusIndexer

        indexer = CorpusIndexer(temp_dir)

        text = "First paragraph with enough words to be meaningful. " * 5 + "\n\n" + \
               "Second paragraph also with sufficient content here. " * 5

        chunks = indexer._split_into_chunks(text, min_words=20, max_words=100)

        assert len(chunks) >= 1
        for chunk in chunks:
            words = len(chunk.split())
            assert words >= 20 or words == len(text.split())  # Short text exception

    def test_split_into_chunks_respects_max(self, temp_dir):
        """Test that chunks respect maximum word limit."""
        from src.rag.corpus_indexer import CorpusIndexer

        indexer = CorpusIndexer(temp_dir)

        # Create text with multiple paragraphs that would exceed max if combined
        paragraphs = ["This is paragraph number one. " * 10 for _ in range(10)]
        text = "\n\n".join(paragraphs)  # ~100 words per paragraph

        chunks = indexer._split_into_chunks(text, min_words=30, max_words=150)

        # Should be split into multiple chunks
        assert len(chunks) > 1

    def test_chunk_id_deterministic(self, temp_dir):
        """Test that chunk IDs are deterministic."""
        from src.rag.corpus_indexer import CorpusIndexer

        indexer = CorpusIndexer(temp_dir)

        id1 = indexer._chunk_id("Author", "source.txt", 0)
        id2 = indexer._chunk_id("Author", "source.txt", 0)

        assert id1 == id2

    def test_chunk_id_unique_for_different_inputs(self, temp_dir):
        """Test that different inputs produce different IDs."""
        from src.rag.corpus_indexer import CorpusIndexer

        indexer = CorpusIndexer(temp_dir)

        id1 = indexer._chunk_id("Author1", "source.txt", 0)
        id2 = indexer._chunk_id("Author2", "source.txt", 0)
        id3 = indexer._chunk_id("Author1", "source.txt", 1)

        assert id1 != id2
        assert id1 != id3

    @patch('src.rag.corpus_indexer.get_embedding_model')
    @patch('src.rag.corpus_indexer.get_chromadb')
    def test_index_corpus_creates_chunks(self, mock_chromadb, mock_embedding, temp_dir, sample_corpus):
        """Test that index_corpus creates and stores chunks."""
        from src.rag.corpus_indexer import CorpusIndexer
        import numpy as np

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(10, 384)  # 10 chunks, 384 dims
        mock_embedding.return_value = mock_model

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value.PersistentClient.return_value = mock_client

        indexer = CorpusIndexer(temp_dir)
        count = indexer.index_corpus(sample_corpus, "Carl Sagan")

        assert count > 0
        mock_collection.upsert.assert_called_once()

    def test_index_corpus_file_not_found(self, temp_dir):
        """Test that missing corpus file raises error."""
        from src.rag.corpus_indexer import CorpusIndexer

        indexer = CorpusIndexer(temp_dir)

        with pytest.raises(FileNotFoundError):
            indexer.index_corpus("/nonexistent/path.txt", "Author")

    @patch('src.rag.corpus_indexer.get_embedding_model')
    @patch('src.rag.corpus_indexer.get_chromadb')
    def test_get_authors(self, mock_chromadb, mock_embedding, temp_dir):
        """Test retrieving list of indexed authors."""
        from src.rag.corpus_indexer import CorpusIndexer

        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "metadatas": [
                {"author": "Sagan"},
                {"author": "Lovecraft"},
                {"author": "Sagan"},
            ]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.return_value.PersistentClient.return_value = mock_client

        indexer = CorpusIndexer(temp_dir)
        authors = indexer.get_authors()

        assert "Sagan" in authors
        assert "Lovecraft" in authors
        assert len(authors) == 2  # Deduplicated


# =============================================================================
# Tests for StyleRetriever
# =============================================================================

class TestStyleRetriever:
    """Tests for StyleRetriever class."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a mock indexer."""
        indexer = MagicMock()
        indexer.get_chunk_count.return_value = 100

        # Mock collection query results
        indexer.collection.query.return_value = {
            "documents": [["Doc 1", "Doc 2", "Doc 3"]],
            "metadatas": [[
                {"author": "Test", "avg_sentence_length": 15.0, "sentence_length_std": 5.0,
                 "dependency_depth": 3.0, "adjective_ratio": 0.1, "verb_ratio": 0.2,
                 "punctuation_density": 2.0, "avg_word_length": 5.0},
                {"author": "Test", "avg_sentence_length": 20.0, "sentence_length_std": 8.0,
                 "dependency_depth": 4.0, "adjective_ratio": 0.15, "verb_ratio": 0.25,
                 "punctuation_density": 3.0, "avg_word_length": 6.0},
                {"author": "Test", "avg_sentence_length": 10.0, "sentence_length_std": 3.0,
                 "dependency_depth": 2.0, "adjective_ratio": 0.08, "verb_ratio": 0.18,
                 "punctuation_density": 1.5, "avg_word_length": 4.5},
            ]],
            "distances": [[0.1, 0.3, 0.5]],
        }

        return indexer

    def test_retrieve_returns_chunks(self, mock_indexer):
        """Test that retrieve returns RetrievedChunk objects."""
        from src.rag.style_retriever import StyleRetriever

        # Mock embedding model
        with patch('src.rag.corpus_indexer.get_embedding_model') as mock_embed:
            import numpy as np
            mock_embed.return_value.encode.return_value = [np.random.rand(384)]

            retriever = StyleRetriever(indexer=mock_indexer)
            results = retriever.retrieve("Test input text.", "Test", k=3)

        assert len(results) == 3
        assert results[0].text == "Doc 1"
        assert results[0].author == "Test"

    def test_retrieve_empty_for_unknown_author(self, mock_indexer):
        """Test that unknown author returns empty results."""
        from src.rag.style_retriever import StyleRetriever

        mock_indexer.get_chunk_count.return_value = 0

        retriever = StyleRetriever(indexer=mock_indexer)
        results = retriever.retrieve("Test input.", "UnknownAuthor", k=3)

        assert len(results) == 0

    def test_retrieve_sorts_by_combined_score(self, mock_indexer):
        """Test that results are sorted by combined score."""
        from src.rag.style_retriever import StyleRetriever

        with patch('src.rag.corpus_indexer.get_embedding_model') as mock_embed:
            import numpy as np
            mock_embed.return_value.encode.return_value = [np.random.rand(384)]

            retriever = StyleRetriever(indexer=mock_indexer)
            results = retriever.retrieve("Test input.", "Test", k=3)

        # Results should be sorted by combined_score descending
        scores = [r.combined_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_respects_k_limit(self, mock_indexer):
        """Test that retrieve returns at most k results."""
        from src.rag.style_retriever import StyleRetriever

        with patch('src.rag.corpus_indexer.get_embedding_model') as mock_embed:
            import numpy as np
            mock_embed.return_value.encode.return_value = [np.random.rand(384)]

            retriever = StyleRetriever(indexer=mock_indexer)
            results = retriever.retrieve("Test input.", "Test", k=2)

        assert len(results) <= 2

    def test_semantic_structural_weights(self):
        """Test that semantic and structural weights are configurable."""
        from src.rag.style_retriever import StyleRetriever

        retriever = StyleRetriever(
            semantic_weight=0.8,
            structural_weight=0.2,
        )

        assert retriever.semantic_weight == 0.8
        assert retriever.structural_weight == 0.2


# =============================================================================
# Tests for SessionContext
# =============================================================================

class TestStyleRAGContext:
    """Tests for StyleRAGContext class."""

    def test_format_for_prompt_empty(self):
        """Test formatting with no examples."""
        from src.rag.session_context import StyleRAGContext

        context = StyleRAGContext(author="Test")

        result = context.format_for_prompt()

        assert result == ""

    def test_format_for_prompt_with_examples(self):
        """Test formatting with examples."""
        from src.rag.session_context import StyleRAGContext

        context = StyleRAGContext(author="Test")
        context.examples = [
            "First example text here.",
            "Second example with more content.",
        ]

        result = context.format_for_prompt()

        assert '[Example 1]: "First example text here."' in result
        assert '[Example 2]: "Second example with more content."' in result

    def test_format_for_prompt_truncates_long_examples(self):
        """Test that long examples are truncated."""
        from src.rag.session_context import StyleRAGContext

        context = StyleRAGContext(author="Test")
        context.examples = ["Word " * 200]  # Very long

        result = context.format_for_prompt()

        # Should be truncated with ...
        assert "..." in result
        assert len(result) < 600

    def test_has_examples(self):
        """Test has_examples method."""
        from src.rag.session_context import StyleRAGContext

        context = StyleRAGContext(author="Test")
        assert context.has_examples() is False

        context.examples = ["Example"]
        assert context.has_examples() is True

    def test_example_count(self):
        """Test example_count property."""
        from src.rag.session_context import StyleRAGContext

        context = StyleRAGContext(author="Test")
        assert context.example_count == 0

        context.examples = ["A", "B", "C"]
        assert context.example_count == 3

    @patch('src.rag.session_context.get_retriever')
    def test_load_examples(self, mock_get_retriever):
        """Test loading examples from retriever."""
        from src.rag.session_context import StyleRAGContext
        from src.rag.style_retriever import RetrievedChunk

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve_diverse.return_value = [
            RetrievedChunk("Example 1", "Test", 0.9, 0.8, 0.85),
            RetrievedChunk("Example 2", "Test", 0.8, 0.7, 0.75),
        ]
        mock_get_retriever.return_value = mock_retriever

        context = StyleRAGContext(author="Test", num_examples=2)
        count = context.load_examples("Sample text to match.")

        assert count == 2
        assert len(context.examples) == 2
        assert context.examples[0] == "Example 1"


class TestRAGContextManager:
    """Tests for RAGContextManager class."""

    @patch('src.rag.session_context.get_retriever')
    def test_get_context_creates_new(self, mock_get_retriever):
        """Test that get_context creates new context."""
        from src.rag.session_context import RAGContextManager

        mock_get_retriever.return_value = MagicMock()

        manager = RAGContextManager()
        context = manager.get_context("Author1")

        assert context.author == "Author1"

    @patch('src.rag.session_context.get_retriever')
    def test_get_context_returns_cached(self, mock_get_retriever):
        """Test that get_context returns cached context."""
        from src.rag.session_context import RAGContextManager

        mock_get_retriever.return_value = MagicMock()

        manager = RAGContextManager()
        context1 = manager.get_context("Author1")
        context2 = manager.get_context("Author1")

        assert context1 is context2

    @patch('src.rag.session_context.get_retriever')
    def test_clear_context(self, mock_get_retriever):
        """Test clearing context for an author."""
        from src.rag.session_context import RAGContextManager

        mock_get_retriever.return_value = MagicMock()

        manager = RAGContextManager()
        manager.get_context("Author1")
        manager.clear_context("Author1")

        # Getting again should create new
        context = manager.get_context("Author1")
        assert context is not None

    @patch('src.rag.session_context.get_retriever')
    def test_clear_all(self, mock_get_retriever):
        """Test clearing all contexts."""
        from src.rag.session_context import RAGContextManager

        mock_get_retriever.return_value = MagicMock()

        manager = RAGContextManager()
        manager.get_context("Author1")
        manager.get_context("Author2")
        manager.clear_all()

        assert len(manager._contexts) == 0


class TestCreateRAGContext:
    """Tests for create_rag_context convenience function."""

    @patch('src.rag.session_context.get_context_manager')
    def test_create_rag_context(self, mock_get_manager):
        """Test the convenience function."""
        from src.rag.session_context import create_rag_context, StyleRAGContext

        mock_manager = MagicMock()
        mock_manager.get_context.return_value = StyleRAGContext(author="Test")
        mock_get_manager.return_value = mock_manager

        context = create_rag_context("Test", sample_text="Hello", num_examples=5)

        mock_manager.get_context.assert_called_once_with(
            author="Test",
            sample_text="Hello",
            num_examples=5,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
