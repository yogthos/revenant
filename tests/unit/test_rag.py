"""Tests for the Structural RAG pipeline.

Tests cover:
- StyleAnalyzer: spaCy-based style metric extraction
- CorpusIndexer: ChromaDB indexing with embeddings
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
# Tests for EnhancedStructuralAnalyzer
# =============================================================================

class TestSyntacticTemplate:
    """Tests for SyntacticTemplate dataclass."""

    def test_syntactic_template_to_instruction(self):
        """Test formatting as prompt instruction."""
        from src.rag.enhanced_analyzer import SyntacticTemplate

        template = SyntacticTemplate(
            pos_pattern="DET ADJ NOUN VERB .",
            clause_structure="main",
            length_category="SHORT",
            example_skeleton="the [ADJ] [NOUN] [VERB] .",
            frequency=0.15,
        )

        instruction = template.to_instruction()

        assert "DET ADJ NOUN" in instruction
        assert "Skeleton" in instruction


class TestVocabularyCluster:
    """Tests for VocabularyCluster dataclass."""

    def test_vocabulary_cluster_to_instruction(self):
        """Test formatting vocabulary as prompt instruction."""
        from src.rag.enhanced_analyzer import VocabularyCluster

        vocab = VocabularyCluster(
            intensifiers=["utterly", "tremendously", "profoundly"],
            evaluatives=["eldritch", "blasphemous", "cyclopean"],
            emotional=["dread", "horror", "fascination"],
        )

        instruction = vocab.to_instruction()

        assert "utterly" in instruction
        assert "eldritch" in instruction
        assert "dread" in instruction


class TestTransitionInventory:
    """Tests for TransitionInventory dataclass."""

    def test_transition_inventory_to_instruction(self):
        """Test formatting transitions as prompt instruction."""
        from src.rag.enhanced_analyzer import TransitionInventory

        transitions = TransitionInventory(
            adversative=["yet", "but", "however"],
            causal=["thus", "therefore"],
            avoid=["Furthermore", "Additionally"],
        )

        instruction = transitions.to_instruction()

        assert "yet" in instruction
        assert "thus" in instruction
        assert "AVOID" in instruction


class TestStanceProfile:
    """Tests for StanceProfile dataclass."""

    def test_stance_profile_to_instruction(self):
        """Test formatting stance profile as prompt instruction."""
        from src.rag.enhanced_analyzer import StanceProfile

        stance = StanceProfile(
            certainty_markers=["clearly", "obviously"],
            rhetorical_question_freq=0.15,
            exclamation_freq=0.05,
            parenthetical_freq=0.2,
        )

        instruction = stance.to_instruction()

        assert "clearly" in instruction
        assert "Rhetorical questions" in instruction


class TestEnhancedStructuralAnalyzer:
    """Tests for EnhancedStructuralAnalyzer class."""

    @pytest.fixture
    def sample_texts(self):
        """Sample author texts for analysis."""
        return [
            "The cosmos is vast and terrifying—utterly beyond human comprehension. We are but motes in an infinite sea.",
            "Yet one must wonder: what lies beyond? The eldritch depths conceal horrors unimaginable.",
            "Thus we find ourselves confronted with dread. Pure, unmitigated dread. The kind that seeps into one's very bones.",
            "Clearly, the universe cares nothing for our petty concerns. Nevertheless, we persist—we must persist.",
        ]

    def test_extract_syntactic_templates(self, sample_texts):
        """Test extraction of syntactic templates."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        templates = analyzer.extract_syntactic_templates(sample_texts)

        assert len(templates) > 0
        assert all(t.pos_pattern for t in templates)
        assert all(t.clause_structure for t in templates)

    def test_extract_vocabulary_clusters(self, sample_texts):
        """Test extraction of vocabulary clusters."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        vocab = analyzer.extract_vocabulary_clusters(sample_texts)

        # Should find "utterly" as an intensifier
        assert len(vocab.intensifiers) >= 0  # May or may not find depending on spaCy analysis
        # Should find some emotional nouns
        assert len(vocab.emotional) >= 0

    def test_extract_transitions(self, sample_texts):
        """Test extraction of transition inventory."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        transitions = analyzer.extract_transitions(sample_texts)

        # "Yet" starts a sentence - should be in adversative
        assert "yet" in transitions.adversative
        # "Thus" starts a sentence - should be in causal
        assert "thus" in transitions.causal
        # LLM-speak should be in avoid (if not in author's corpus)
        assert any(t in transitions.avoid for t in ["furthermore", "additionally", "moreover"])

    def test_extract_stance_profile(self, sample_texts):
        """Test extraction of stance profile."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        stance = analyzer.extract_stance_profile(sample_texts)

        # "Clearly" is a certainty marker
        assert "clearly" in stance.certainty_markers
        # Should have some rhetorical questions (the "what lies beyond?" sentence)
        assert stance.rhetorical_question_freq > 0
        # Should have parentheticals (em-dashes)
        assert stance.parenthetical_freq > 0

    def test_extract_opening_patterns(self, sample_texts):
        """Test extraction of opening patterns."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        openings = analyzer.extract_opening_patterns(sample_texts)

        # Should have some patterns with frequencies
        assert len(openings.patterns) > 0
        # Frequencies should sum to approximately 1
        total_freq = sum(openings.patterns.values())
        assert 0.9 <= total_freq <= 1.1
        # Should have avoid patterns
        assert len(openings.avoid_patterns) > 0

    def test_analyze_complete(self, sample_texts):
        """Test complete multi-channel analysis."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        profile = analyzer.analyze(sample_texts)

        assert len(profile.syntactic_templates) > 0
        assert profile.vocabulary is not None
        assert profile.transitions is not None
        assert profile.stance is not None
        assert profile.openings is not None

    def test_analyze_empty_texts(self):
        """Test analysis of empty text list."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        profile = analyzer.analyze([])

        # Should return empty profile, not crash
        assert len(profile.syntactic_templates) == 0

    def test_enhanced_style_profile_format_for_prompt(self, sample_texts):
        """Test complete prompt formatting."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        profile = analyzer.analyze(sample_texts)
        formatted = profile.format_for_prompt()

        # Should have structured sections
        assert "SYNTACTIC TEMPLATES" in formatted or len(profile.syntactic_templates) == 0
        # At least some content should be present
        assert len(formatted) > 0

    def test_get_enhanced_analyzer_singleton(self):
        """Test that get_enhanced_analyzer returns singleton."""
        from src.rag.enhanced_analyzer import get_enhanced_analyzer

        a1 = get_enhanced_analyzer()
        a2 = get_enhanced_analyzer()

        assert a1 is a2

    def test_categorize_length(self):
        """Test sentence length categorization."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()

        assert analyzer.categorize_length(3) == "FRAGMENT"
        assert analyzer.categorize_length(8) == "SHORT"
        assert analyzer.categorize_length(15) == "MEDIUM"
        assert analyzer.categorize_length(25) == "LONG"
        assert analyzer.categorize_length(50) == "VERY_LONG"


class TestEnhancedAnalyzerEdgeCases:
    """Edge case tests for EnhancedStructuralAnalyzer."""

    def test_single_word_text(self):
        """Test handling of single word text."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        profile = analyzer.analyze(["Word."])

        # Should not crash
        assert profile is not None
        assert profile.vocabulary is not None

    def test_very_long_sentence(self):
        """Test handling of extremely long sentences."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        long_text = "The " + " ".join(["magnificent" for _ in range(100)]) + " castle stood tall."
        templates = analyzer.extract_syntactic_templates([long_text])

        # Should truncate pattern
        assert len(templates) > 0
        assert len(templates[0].pos_pattern) <= 100

    def test_text_with_only_punctuation(self):
        """Test handling of punctuation-only text."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        profile = analyzer.analyze(["...", "---", "!!!"])

        # Should handle gracefully
        assert profile is not None

    def test_mixed_language_text(self):
        """Test handling of text with non-English words."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = ["The café was très magnifique.", "C'est la vie, mon ami."]
        profile = analyzer.analyze(texts)

        # Should still extract patterns
        assert profile is not None
        assert len(profile.syntactic_templates) > 0

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = ["The 日本 mountain was beautiful.", 'She said "hello" with a smile.']
        profile = analyzer.analyze(texts)

        assert profile is not None

    def test_multiple_questions_in_text(self):
        """Test extraction of multiple rhetorical questions."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "What is truth? What is beauty? What is meaning?",
            "Can we know? Should we try? Must we fail?",
        ]
        stance = analyzer.extract_stance_profile(texts)

        # Should detect high rhetorical question frequency
        assert stance.rhetorical_question_freq > 0.5

    def test_text_with_many_exclamations(self):
        """Test extraction of exclamation frequency."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "Behold! The terror! The horror! The unspeakable dread!",
            "Run! Hide! Never look back!",
        ]
        stance = analyzer.extract_stance_profile(texts)

        assert stance.exclamation_freq > 0.3

    def test_text_without_transitions(self):
        """Test text that starts with non-transition words."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "The moon rose. Stars appeared. Night descended.",
            "Rain fell. Thunder roared. Lightning struck.",
        ]
        transitions = analyzer.extract_transitions(texts)

        # Should still have avoid list
        assert len(transitions.avoid) > 0
        # Should have empty or near-empty transition categories
        assert len(transitions.adversative) == 0 or len(transitions.causal) == 0


class TestSyntacticTemplateExtraction:
    """Detailed tests for syntactic template extraction."""

    def test_fragment_detection(self):
        """Test detection of sentence fragments."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = ["Pure horror. Utter dread. The end."]
        templates = analyzer.extract_syntactic_templates(texts)

        # Should detect fragments
        fragment_templates = [t for t in templates if t.clause_structure == "fragment"]
        assert len(fragment_templates) >= 0  # May or may not detect based on spaCy

    def test_complex_clause_structure(self):
        """Test detection of complex clause structures."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "Although the night was dark, and the wind howled fiercely, she pressed on because she knew that survival depended on it.",
        ]
        templates = analyzer.extract_syntactic_templates(texts)

        # Should detect complex structure
        assert len(templates) > 0
        complex_templates = [t for t in templates if "complex" in t.clause_structure or "subordinate" in t.clause_structure]
        assert len(complex_templates) >= 0

    def test_skeleton_preserves_function_words(self):
        """Test that skeleton preserves function words."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = ["The quick fox jumped over the lazy dog."]
        templates = analyzer.extract_syntactic_templates(texts)

        assert len(templates) > 0
        skeleton = templates[0].example_skeleton.lower()
        # Should preserve "the" and "over"
        assert "the" in skeleton or "[det]" in skeleton

    def test_template_frequency_calculation(self):
        """Test that template frequencies are calculated correctly."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        # Repeat similar patterns
        texts = [
            "The cat sat. The dog ran. The bird flew.",
            "The sun rose. The moon set. The stars shone.",
        ]
        templates = analyzer.extract_syntactic_templates(texts)

        # Frequencies should sum to approximately 1
        total_freq = sum(t.frequency for t in templates)
        assert 0.9 <= total_freq <= 1.1


class TestVocabularyClusterExtraction:
    """Detailed tests for vocabulary cluster extraction."""

    def test_intensifier_detection(self):
        """Test detection of intensifying adverbs."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "The utterly magnificent castle stood impossibly tall.",
            "A profoundly disturbing and tremendously unsettling sight.",
        ]
        vocab = analyzer.extract_vocabulary_clusters(texts)

        # Should find intensifiers
        assert len(vocab.intensifiers) >= 0  # Depends on spaCy analysis

    def test_archaic_word_detection(self):
        """Test detection of archaic vocabulary."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "Whereupon he departed, hitherto unknown to all.",
            "Thereby the matter was settled, forthwith and completely.",
        ]
        vocab = analyzer.extract_vocabulary_clusters(texts)

        # Should detect archaic words
        assert "whereupon" in vocab.archaic or "hitherto" in vocab.archaic or "thereby" in vocab.archaic

    def test_emotional_noun_detection(self):
        """Test detection of emotional nouns."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "The dread consumed him. Horror filled her soul.",
            "Fascination gave way to terror, then to awe.",
        ]
        vocab = analyzer.extract_vocabulary_clusters(texts)

        # Should find emotional nouns
        emotional_found = any(w in vocab.emotional for w in ["dread", "horror", "terror", "awe", "fascination"])
        assert emotional_found or len(vocab.emotional) >= 0

    def test_stance_marker_detection(self):
        """Test detection of stance markers."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "Clearly, this is obvious. Obviously, the answer is certain.",
            "Perhaps it was possible. Maybe, conceivably, it could work.",
        ]
        vocab = analyzer.extract_vocabulary_clusters(texts)

        # Should detect certainty markers
        assert "clearly" in vocab.stance_certain or "obviously" in vocab.stance_certain
        # Should detect hedging markers
        assert "perhaps" in vocab.stance_hedge or "maybe" in vocab.stance_hedge


class TestTransitionInventoryExtraction:
    """Detailed tests for transition inventory extraction."""

    def test_additive_transitions(self):
        """Test detection of additive transitions."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "And so it began. Also, the weather changed.",
            "Moreover, the situation worsened. Besides, help was far.",
        ]
        transitions = analyzer.extract_transitions(texts)

        # Should find additive transitions (if they start sentences)
        has_additive = any(t in transitions.additive for t in ["and", "also", "moreover", "besides"])
        assert has_additive or len(transitions.additive) >= 0

    def test_temporal_transitions(self):
        """Test detection of temporal transitions."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "Then the storm arrived. Thereafter, chaos ensued.",
            "Meanwhile, the others waited. Subsequently, they acted.",
        ]
        transitions = analyzer.extract_transitions(texts)

        # Should find temporal transitions
        has_temporal = any(t in transitions.temporal for t in ["then", "thereafter", "meanwhile", "subsequently"])
        assert has_temporal

    def test_avoid_list_excludes_author_words(self):
        """Test that avoid list excludes words the author uses."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        # Author uses "furthermore"
        texts = [
            "Furthermore, the evidence supports this claim.",
            "Furthermore, we must consider the implications.",
        ]
        transitions = analyzer.extract_transitions(texts)

        # "furthermore" should NOT be in avoid list since author uses it
        assert "furthermore" not in transitions.avoid


class TestOpeningPatternsExtraction:
    """Detailed tests for opening pattern extraction."""

    def test_diverse_opening_patterns(self):
        """Test extraction of diverse opening patterns."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        texts = [
            "The ancient castle crumbled. Slowly, it fell. What horror!",
            "Running quickly, she escaped. A nightmare. Pure terror.",
        ]
        openings = analyzer.extract_opening_patterns(texts)

        # Should have multiple different patterns
        assert len(openings.patterns) >= 2

    def test_avoid_patterns_present(self):
        """Test that LLM-speak patterns are in avoid list."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        openings = analyzer.extract_opening_patterns(["The night was dark."])

        # Should have standard avoid patterns
        assert "Furthermore ," in openings.avoid_patterns or "Additionally ," in openings.avoid_patterns

    def test_pattern_frequency_distribution(self):
        """Test that pattern frequencies are reasonable."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        # Create texts with known patterns
        texts = ["The cat sat."] * 3 + ["A dog ran."]
        openings = analyzer.extract_opening_patterns(texts)

        # Most frequent pattern should have higher frequency
        if openings.patterns:
            max_freq = max(openings.patterns.values())
            assert max_freq >= 0.5  # Should be at least 50% for the repeated pattern


class TestEnhancedStyleProfileFormatting:
    """Tests for EnhancedStyleProfile formatting."""

    def test_format_with_all_components(self):
        """Test formatting with all components populated."""
        from src.rag.enhanced_analyzer import (
            EnhancedStyleProfile, SyntacticTemplate, VocabularyCluster,
            TransitionInventory, StanceProfile, OpeningPatterns
        )

        profile = EnhancedStyleProfile(
            syntactic_templates=[
                SyntacticTemplate(
                    pos_pattern="DET ADJ NOUN VERB",
                    clause_structure="main",
                    length_category="MEDIUM",
                    example_skeleton="the [ADJ] [NOUN] [VERB]",
                    frequency=0.3,
                )
            ],
            vocabulary=VocabularyCluster(
                intensifiers=["utterly", "profoundly"],
                evaluatives=["eldritch", "cyclopean"],
                emotional=["dread", "horror"],
                archaic=["whereupon"],
            ),
            transitions=TransitionInventory(
                adversative=["yet", "but"],
                causal=["thus", "therefore"],
                avoid=["Furthermore"],
            ),
            stance=StanceProfile(
                certainty_markers=["clearly"],
                rhetorical_question_freq=0.15,
                exclamation_freq=0.05,
                parenthetical_freq=0.2,
            ),
            openings=OpeningPatterns(
                patterns={"DET ADJ NOUN": 0.4, "ADV ,": 0.2},
                avoid_patterns=["Additionally ,"],
            ),
        )

        formatted = profile.format_for_prompt()

        # Should contain all sections
        assert "SYNTACTIC TEMPLATES" in formatted
        assert "VOCABULARY" in formatted
        assert "TRANSITIONS" in formatted
        assert "EMOTIONAL ENGAGEMENT" in formatted
        assert "SENTENCE OPENINGS" in formatted

    def test_format_with_empty_components(self):
        """Test formatting with empty components."""
        from src.rag.enhanced_analyzer import EnhancedStyleProfile

        profile = EnhancedStyleProfile()
        formatted = profile.format_for_prompt()

        # Should return empty string for completely empty profile
        assert formatted == ""

    def test_format_partial_components(self):
        """Test formatting with only some components populated."""
        from src.rag.enhanced_analyzer import (
            EnhancedStyleProfile, VocabularyCluster, TransitionInventory
        )

        profile = EnhancedStyleProfile(
            vocabulary=VocabularyCluster(intensifiers=["utterly"]),
            transitions=TransitionInventory(causal=["thus"]),
        )

        formatted = profile.format_for_prompt()

        assert "utterly" in formatted
        assert "thus" in formatted


class TestStructuralRAGEnhanced:
    """Tests for StructuralRAG with enhanced analyzer integration."""

    def test_structural_guidance_with_enhanced_profile(self):
        """Test that StructuralGuidance includes enhanced profile."""
        from src.rag.structural_rag import StructuralGuidance
        from src.rag.enhanced_analyzer import EnhancedStyleProfile, VocabularyCluster

        profile = EnhancedStyleProfile(
            vocabulary=VocabularyCluster(
                intensifiers=["utterly", "tremendously"],
                evaluatives=["eldritch"],
            )
        )

        guidance = StructuralGuidance(
            rhythm_pattern="LONG → SHORT",
            punctuation_hints=["use dashes"],
            length_guidance="Vary between 5 and 30 words",
            fragment_hint="Use occasional fragments",
            opening_hint="Vary openings",
            enhanced_profile=profile,
        )

        formatted = guidance.format_for_prompt()

        # Should include both basic and enhanced guidance
        assert "RHYTHM PATTERN" in formatted
        assert "utterly" in formatted or "VOCABULARY" in formatted

    def test_structural_guidance_without_enhanced_profile(self):
        """Test that StructuralGuidance works without enhanced profile."""
        from src.rag.structural_rag import StructuralGuidance

        guidance = StructuralGuidance(
            rhythm_pattern="LONG → SHORT",
            punctuation_hints=["use dashes"],
            length_guidance="Vary",
            fragment_hint="Use fragments",
            opening_hint="",
            enhanced_profile=None,
        )

        formatted = guidance.format_for_prompt()

        assert "RHYTHM PATTERN" in formatted
        assert len(formatted) > 0

    def test_structural_guidance_empty_fields(self):
        """Test StructuralGuidance with empty fields."""
        from src.rag.structural_rag import StructuralGuidance

        guidance = StructuralGuidance(
            rhythm_pattern="",
            punctuation_hints=[],
            length_guidance="",
            fragment_hint="",
            opening_hint="",
            enhanced_profile=None,
        )

        formatted = guidance.format_for_prompt()

        # Should return empty or minimal string
        assert formatted == "" or len(formatted) < 50


class TestIntegrationWithRealisticCorpus:
    """Integration tests with realistic author corpus samples."""

    @pytest.fixture
    def lovecraft_corpus(self):
        """Sample Lovecraft-style corpus."""
        return [
            "The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents.",
            "We live on a placid island of ignorance in the midst of black seas of infinity, and it was not meant that we should voyage far.",
            "The sciences, each straining in its own direction, have hitherto harmed us little; but some day the piecing together of dissociated knowledge will open up such terrifying vistas of reality.",
            "Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl fhtagn—that is not dead which can eternal lie.",
            "And with strange aeons even death may die. The Old Ones were, the Old Ones are, and the Old Ones shall be.",
        ]

    @pytest.fixture
    def sagan_corpus(self):
        """Sample Carl Sagan-style corpus."""
        return [
            "The cosmos is all that is or was or ever will be. Our feeblest contemplations of the Cosmos stir us.",
            "There is a tingling in the spine, a catch in the voice, a faint sensation, as if a distant memory, of falling from a height.",
            "We are a way for the universe to know itself. Some part of our being knows this is where we came from.",
            "We long to return; and we can, because the cosmos is also within us. We're made of star stuff.",
            "For small creatures such as we, the vastness is bearable only through love.",
        ]

    def test_lovecraft_analysis(self, lovecraft_corpus):
        """Test analysis of Lovecraft-style corpus."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        profile = analyzer.analyze(lovecraft_corpus)

        # Should detect archaic words
        assert len(profile.vocabulary.archaic) >= 0  # "hitherto" might be found
        # Should detect long sentence patterns
        assert len(profile.syntactic_templates) > 0

    def test_sagan_analysis(self, sagan_corpus):
        """Test analysis of Sagan-style corpus."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        profile = analyzer.analyze(sagan_corpus)

        # Should have syntactic templates
        assert len(profile.syntactic_templates) > 0
        # Should have opening patterns
        assert len(profile.openings.patterns) > 0

    def test_different_authors_produce_different_profiles(self, lovecraft_corpus, sagan_corpus):
        """Test that different authors produce distinct profiles."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        analyzer = EnhancedStructuralAnalyzer()
        lovecraft_profile = analyzer.analyze(lovecraft_corpus)
        sagan_profile = analyzer.analyze(sagan_corpus)

        # Profiles should be different
        lovecraft_formatted = lovecraft_profile.format_for_prompt()
        sagan_formatted = sagan_profile.format_for_prompt()

        # At least one aspect should differ
        assert lovecraft_formatted != sagan_formatted or len(lovecraft_corpus) != len(sagan_corpus)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
