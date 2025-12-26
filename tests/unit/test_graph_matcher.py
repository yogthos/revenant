"""Unit tests for graph matching."""

import pytest
from unittest.mock import MagicMock, patch

from src.planning.graph_matcher import (
    GraphMatcher,
    MatchedStyleGraph,
    FallbackMatcher,
)
from src.models.graph import (
    SemanticGraph,
    PropositionNode,
    RelationshipEdge,
    RelationshipType,
    ParagraphRole,
    RhetoricalIntent,
)
from src.models.style import AuthorProfile, StyleProfile


class TestMatchedStyleGraph:
    """Test MatchedStyleGraph data class."""

    def test_create_matched_graph(self):
        """Test creating a matched style graph."""
        match = MatchedStyleGraph(
            id="test123",
            text="Sample paragraph text.",
            skeleton="[S1] [S2]",
            intent="ARGUMENT",
            signature="CAUSALITY",
            author="test_author",
            role="BODY",
            burstiness=0.3,
            node_count=3,
            edge_types=["CAUSES", "FOLLOWS"],
            similarity_score=0.85,
            edge_overlap_score=0.7
        )

        assert match.id == "test123"
        assert match.skeleton == "[S1] [S2]"
        assert match.similarity_score == 0.85

    def test_combined_score(self):
        """Test combined score calculation."""
        match = MatchedStyleGraph(
            id="test",
            text="text",
            skeleton="",
            intent="ARGUMENT",
            signature="SEQUENCE",
            author="author",
            role="BODY",
            burstiness=0.0,
            node_count=2,
            edge_types=[],
            similarity_score=0.8,
            edge_overlap_score=0.5
        )

        # Combined = 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.2 = 0.68
        assert match.combined_score == pytest.approx(0.68)


class TestGraphMatcher:
    """Test GraphMatcher functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample semantic graph."""
        nodes = [
            PropositionNode(id="p1", text="First proposition.", subject="First", verb="is"),
            PropositionNode(id="p2", text="Second proposition.", subject="Second", verb="follows"),
        ]
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.FOLLOWS
            )
        ]
        return SemanticGraph(
            nodes=nodes,
            edges=edges,
            paragraph_idx=0,
            role=ParagraphRole.BODY,
            intent=RhetoricalIntent.ARGUMENT
        )

    @pytest.fixture
    def style_profile(self):
        """Create a sample style profile."""
        author = AuthorProfile(
            name="Test Author",
            style_dna="Clear writing style.",
            avg_sentence_length=15.0,
            burstiness=0.3
        )
        return StyleProfile.from_author(author)

    def test_matcher_without_indexer(self, sample_graph, style_profile):
        """Test matcher without indexer returns empty."""
        matcher = GraphMatcher(indexer=None)
        results = matcher.find_matches(sample_graph, style_profile)

        assert results == []

    def test_create_query_text(self, sample_graph):
        """Test query text creation."""
        matcher = GraphMatcher()
        query = matcher._create_query_text(sample_graph)

        assert "First proposition" in query
        assert "Second proposition" in query

    def test_calculate_edge_overlap_identical(self):
        """Test edge overlap with identical edges."""
        matcher = GraphMatcher()
        overlap = matcher._calculate_edge_overlap(
            ["CAUSES", "FOLLOWS"],
            ["CAUSES", "FOLLOWS"]
        )

        assert overlap == 1.0

    def test_calculate_edge_overlap_partial(self):
        """Test edge overlap with partial match."""
        matcher = GraphMatcher()
        overlap = matcher._calculate_edge_overlap(
            ["CAUSES", "FOLLOWS"],
            ["CAUSES", "CONTRASTS"]
        )

        # Jaccard: {CAUSES} / {CAUSES, FOLLOWS, CONTRASTS} = 1/3
        assert overlap == pytest.approx(1/3)

    def test_calculate_edge_overlap_empty(self):
        """Test edge overlap with empty sets."""
        matcher = GraphMatcher()

        # Both empty
        assert matcher._calculate_edge_overlap([], []) == 1.0

        # One empty
        assert matcher._calculate_edge_overlap(["CAUSES"], []) == 0.5
        assert matcher._calculate_edge_overlap([], ["CAUSES"]) == 0.5

    def test_find_best_match_returns_single(self, sample_graph, style_profile):
        """Test find_best_match returns single result."""
        # Mock indexer
        mock_indexer = MagicMock()
        mock_indexer.query_graphs.return_value = [
            {
                "id": "match1",
                "text": "Matched text",
                "distance": 0.2,
                "metadata": {
                    "skeleton": "[S1]",
                    "intent": "ARGUMENT",
                    "signature": "SEQUENCE",
                    "author": "Test Author",
                    "role": "BODY",
                    "burstiness": 0.3,
                    "node_count": 2,
                    "edge_types": '["FOLLOWS"]'
                }
            }
        ]

        matcher = GraphMatcher(indexer=mock_indexer)
        result = matcher.find_best_match(sample_graph, style_profile)

        assert result is not None
        assert result.id == "match1"


class TestFallbackMatcher:
    """Test FallbackMatcher functionality."""

    @pytest.fixture
    def style_profile(self):
        """Create a sample style profile."""
        author = AuthorProfile(
            name="Test Author",
            style_dna="Clear writing style.",
            avg_sentence_length=15.0,
            burstiness=0.3
        )
        return StyleProfile.from_author(author)

    @pytest.fixture
    def sample_graph(self):
        """Create a sample semantic graph."""
        nodes = [
            PropositionNode(id=f"p{i}", text=f"Prop {i}.", subject="S", verb="is")
            for i in range(4)
        ]
        return SemanticGraph(nodes=nodes, edges=[])

    def test_create_default_structure(self, style_profile, sample_graph):
        """Test creating default structure."""
        fallback = FallbackMatcher(style_profile)
        structure = fallback.create_default_structure(sample_graph)

        assert "num_sentences" in structure
        assert "target_lengths" in structure
        assert structure["is_fallback"] is True
        assert structure["skeleton"] is None

    def test_default_structure_sentence_count(self, style_profile, sample_graph):
        """Test sentence count estimation."""
        fallback = FallbackMatcher(style_profile)
        structure = fallback.create_default_structure(sample_graph)

        # With 4 props and avg ~15 words, should estimate 1-2 sentences
        assert structure["num_sentences"] >= 1
        assert structure["num_sentences"] <= 4

    def test_generate_length_pattern_uniform(self, style_profile):
        """Test length pattern with low burstiness."""
        fallback = FallbackMatcher(style_profile)
        lengths = fallback._generate_length_pattern(3, 15.0, 0.05)

        # Low burstiness = uniform lengths
        assert len(lengths) == 3
        assert all(l == 15 for l in lengths)

    def test_generate_length_pattern_single(self, style_profile):
        """Test length pattern with single sentence."""
        fallback = FallbackMatcher(style_profile)
        lengths = fallback._generate_length_pattern(1, 20.0, 0.3)

        assert lengths == [20]
