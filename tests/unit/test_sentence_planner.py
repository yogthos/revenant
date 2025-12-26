"""Unit tests for sentence planning."""

import pytest
from unittest.mock import MagicMock, patch

from src.planning.sentence_planner import (
    SentencePlanner,
    PropositionClusterer,
    RELATIONSHIP_TO_TRANSITION,
)
from src.planning.rhythm_planner import RhythmPattern
from src.models.graph import (
    SemanticGraph,
    PropositionNode,
    RelationshipEdge,
    RelationshipType,
    ParagraphRole,
    RhetoricalIntent,
)
from src.models.plan import SentenceRole, TransitionType
from src.models.style import AuthorProfile, StyleProfile


class TestPropositionClusterer:
    """Test PropositionClusterer functionality."""

    @pytest.fixture
    def clusterer(self):
        """Create a clusterer instance."""
        return PropositionClusterer(words_per_proposition=8.0)

    @pytest.fixture
    def sample_propositions(self):
        """Create sample propositions."""
        return [
            PropositionNode(id="p1", text="First point.", subject="First", verb="is"),
            PropositionNode(id="p2", text="Second point.", subject="Second", verb="follows"),
            PropositionNode(id="p3", text="Third point.", subject="Third", verb="adds"),
            PropositionNode(id="p4", text="Fourth point.", subject="Fourth", verb="concludes"),
        ]

    def test_cluster_empty_propositions(self, clusterer):
        """Test clustering empty propositions."""
        rhythm = RhythmPattern(lengths=[15, 15], burstiness=0.2, avg_length=15.0)
        clusters = clusterer.cluster([], [], rhythm)

        assert clusters == []

    def test_cluster_empty_rhythm(self, clusterer, sample_propositions):
        """Test clustering with empty rhythm pattern."""
        rhythm = RhythmPattern(lengths=[], burstiness=0.0, avg_length=0.0)
        clusters = clusterer.cluster(sample_propositions, [], rhythm)

        # Each proposition gets its own sentence
        assert len(clusters) == 4
        assert all(len(c) == 1 for c in clusters)

    def test_cluster_fewer_props_than_sentences(self, clusterer):
        """Test when fewer propositions than target sentences."""
        props = [
            PropositionNode(id="p1", text="First.", subject="First", verb="is"),
            PropositionNode(id="p2", text="Second.", subject="Second", verb="is"),
        ]
        rhythm = RhythmPattern(lengths=[15, 15, 15, 15], burstiness=0.2, avg_length=15.0)

        clusters = clusterer.cluster(props, [], rhythm)

        # Each prop gets its own cluster
        assert len(clusters) == 2
        assert all(len(c) == 1 for c in clusters)

    def test_cluster_more_props_than_sentences(self, clusterer, sample_propositions):
        """Test when more propositions than target sentences."""
        rhythm = RhythmPattern(lengths=[20, 20], burstiness=0.2, avg_length=20.0)

        clusters = clusterer.cluster(sample_propositions, [], rhythm)

        # Should have 2 clusters with props distributed
        assert len(clusters) == 2
        total_props = sum(len(c) for c in clusters)
        assert total_props == 4

    def test_cluster_all_props_assigned(self, clusterer, sample_propositions):
        """Test that all propositions are assigned to clusters."""
        rhythm = RhythmPattern(lengths=[15, 15, 15], burstiness=0.2, avg_length=15.0)

        clusters = clusterer.cluster(sample_propositions, [], rhythm)

        all_props = []
        for cluster in clusters:
            all_props.extend(cluster)

        assert len(all_props) == 4
        prop_ids = {p.id for p in all_props}
        assert prop_ids == {"p1", "p2", "p3", "p4"}

    def test_build_adjacency(self, clusterer, sample_propositions):
        """Test building adjacency map."""
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.CAUSES
            ),
            RelationshipEdge(
                source_id="p2",
                target_id="p3",
                relationship=RelationshipType.ELABORATES
            ),
        ]

        adjacency = clusterer._build_adjacency(sample_propositions, edges)

        assert "p2" in adjacency["p1"]
        assert "p1" in adjacency["p2"]
        assert "p3" in adjacency["p2"]
        assert "p2" in adjacency["p3"]

    def test_greedy_cluster_distributes_evenly(self, clusterer, sample_propositions):
        """Test greedy clustering distributes propositions."""
        target_lengths = [16, 16]  # 2 words per prop → 2 props each

        clusters = clusterer._greedy_cluster(
            sample_propositions,
            target_lengths,
            {p.id: [] for p in sample_propositions}
        )

        # Should distribute 4 props into 2 clusters
        assert len(clusters) == 2


class TestSentencePlanner:
    """Test SentencePlanner functionality."""

    @pytest.fixture
    def style_profile(self):
        """Create a sample style profile."""
        author = AuthorProfile(
            name="Test Author",
            style_dna="Clear, concise writing.",
            avg_sentence_length=15.0,
            burstiness=0.25
        )
        return StyleProfile.from_author(author)

    @pytest.fixture
    def planner(self, style_profile):
        """Create a sentence planner."""
        return SentencePlanner(style_profile)

    @pytest.fixture
    def sample_graph(self):
        """Create a sample semantic graph."""
        nodes = [
            PropositionNode(id="p1", text="Main point here.", subject="Main", verb="is"),
            PropositionNode(id="p2", text="Supporting detail.", subject="Detail", verb="supports"),
            PropositionNode(id="p3", text="Final conclusion.", subject="Conclusion", verb="follows"),
        ]
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.ELABORATES
            ),
            RelationshipEdge(
                source_id="p2",
                target_id="p3",
                relationship=RelationshipType.FOLLOWS
            ),
        ]
        return SemanticGraph(
            nodes=nodes,
            edges=edges,
            paragraph_idx=0,
            role=ParagraphRole.BODY,
            intent=RhetoricalIntent.ARGUMENT
        )

    def test_create_plan_empty_graph(self, planner):
        """Test creating plan from empty graph."""
        graph = SemanticGraph(
            nodes=[],
            edges=[],
            role=ParagraphRole.BODY,
            intent=RhetoricalIntent.ARGUMENT
        )

        plan = planner.create_plan(graph)

        assert len(plan.nodes) == 0
        assert plan.paragraph_intent == "ARGUMENT"
        assert plan.paragraph_role == "BODY"

    def test_create_plan_basic(self, planner, sample_graph):
        """Test creating a basic sentence plan."""
        plan = planner.create_plan(sample_graph)

        assert len(plan.nodes) >= 1
        assert plan.paragraph_intent == "ARGUMENT"
        assert plan.paragraph_role == "BODY"
        assert plan.source_graph == sample_graph

    def test_create_plan_with_matcher(self, style_profile, sample_graph):
        """Test creating plan with graph matcher."""
        mock_matcher = MagicMock()
        mock_matcher.find_best_match.return_value = MagicMock(
            id="match123",
            skeleton="[S1] [S2]"
        )

        planner = SentencePlanner(style_profile, graph_matcher=mock_matcher)
        plan = planner.create_plan(sample_graph)

        mock_matcher.find_best_match.assert_called_once()
        assert plan.matched_style_graph_id == "match123"

    def test_create_plan_propositions_distributed(self, planner, sample_graph):
        """Test that all propositions are distributed to sentences."""
        plan = planner.create_plan(sample_graph)

        all_props = []
        for node in plan.nodes:
            all_props.extend(node.propositions)

        assert len(all_props) == 3

    def test_determine_role_first_sentence_intro(self, planner):
        """Test role determination for first sentence in intro."""
        graph = SemanticGraph(
            nodes=[],
            edges=[],
            role=ParagraphRole.INTRO
        )

        role = planner._determine_role(0, 3, graph)
        assert role == SentenceRole.THESIS

    def test_determine_role_first_sentence_body(self, planner):
        """Test role determination for first sentence in body."""
        graph = SemanticGraph(
            nodes=[],
            edges=[],
            role=ParagraphRole.BODY
        )

        role = planner._determine_role(0, 4, graph)
        assert role == SentenceRole.THESIS

    def test_determine_role_last_sentence(self, planner):
        """Test role determination for last sentence."""
        graph = SemanticGraph(
            nodes=[],
            edges=[],
            role=ParagraphRole.BODY
        )

        role = planner._determine_role(3, 4, graph)
        assert role == SentenceRole.CONCLUSION

    def test_determine_role_middle_sentence(self, planner):
        """Test role determination for middle sentence."""
        graph = SemanticGraph(
            nodes=[],
            edges=[],
            role=ParagraphRole.BODY
        )

        role = planner._determine_role(1, 4, graph)
        assert role == SentenceRole.ELABORATION

    def test_determine_transition_first_sentence(self, planner):
        """Test transition for first sentence is NONE."""
        cluster = [PropositionNode(id="p1", text="Text.", subject="S", verb="is")]

        transition = planner._determine_transition(0, cluster, [])
        assert transition == TransitionType.NONE

    def test_determine_transition_causal(self, planner):
        """Test causal transition detection."""
        cluster = [PropositionNode(id="p2", text="Text.", subject="S", verb="is")]
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.CAUSES
            )
        ]

        transition = planner._determine_transition(1, cluster, edges)
        assert transition == TransitionType.CAUSAL

    def test_determine_transition_adversative(self, planner):
        """Test adversative transition detection."""
        cluster = [PropositionNode(id="p2", text="Text.", subject="S", verb="is")]
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.CONTRASTS
            )
        ]

        transition = planner._determine_transition(1, cluster, edges)
        assert transition == TransitionType.ADVERSATIVE

    def test_detect_signature_empty(self, planner):
        """Test signature detection with no edges."""
        signature = planner._detect_signature([])
        assert signature == "SEQUENCE"

    def test_detect_signature_contrast(self, planner):
        """Test signature detection with contrast edges."""
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.CONTRASTS
            ),
            RelationshipEdge(
                source_id="p2",
                target_id="p3",
                relationship=RelationshipType.CONTRASTS
            ),
        ]

        signature = planner._detect_signature(edges)
        assert signature == "CONTRAST"

    def test_detect_signature_causality(self, planner):
        """Test signature detection with causal edges."""
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.CAUSES
            ),
        ]

        signature = planner._detect_signature(edges)
        assert signature == "CAUSALITY"

    def test_detect_signature_elaboration(self, planner):
        """Test signature detection with elaboration edges."""
        edges = [
            RelationshipEdge(
                source_id="p1",
                target_id="p2",
                relationship=RelationshipType.ELABORATES
            ),
            RelationshipEdge(
                source_id="p2",
                target_id="p3",
                relationship=RelationshipType.ELABORATES
            ),
        ]

        signature = planner._detect_signature(edges)
        assert signature == "ELABORATION"

    def test_extract_skeleton_segment_found(self, planner):
        """Test skeleton segment extraction when found."""
        skeleton = "[S1] introduces the concept. [S2] provides evidence. [S3] concludes."

        segment = planner._extract_skeleton_segment(skeleton, 1, 3)
        assert segment == "[S2]"

    def test_extract_skeleton_segment_not_found(self, planner):
        """Test skeleton segment extraction when not found."""
        skeleton = "[S1] [S2]"

        segment = planner._extract_skeleton_segment(skeleton, 5, 6)
        assert segment is None

    def test_extract_keywords(self, planner):
        """Test keyword extraction from propositions."""
        props = [
            PropositionNode(
                id="p1",
                text="Evolution drives change.",
                subject="Evolution",
                verb="drives",
                entities=["Darwin", "natural selection"]
            ),
            PropositionNode(
                id="p2",
                text="Species adapt.",
                subject="Species",
                verb="adapt",
                entities=["adaptation"]
            ),
        ]

        keywords = planner._extract_keywords(props)

        # Should include some entities and subjects
        assert len(keywords) <= 5
        # Should include entities from propositions
        assert any(k in ["Darwin", "natural selection", "adaptation", "Evolution", "Species"] for k in keywords)

    def test_sentence_nodes_have_target_lengths(self, planner, sample_graph):
        """Test that sentence nodes have target lengths."""
        plan = planner.create_plan(sample_graph)

        for node in plan.nodes:
            assert node.target_length >= 5  # Minimum length

    def test_sentence_nodes_have_ids(self, planner, sample_graph):
        """Test that sentence nodes have unique IDs."""
        plan = planner.create_plan(sample_graph)

        ids = [node.id for node in plan.nodes]
        assert len(ids) == len(set(ids))  # All unique


class TestRelationshipToTransition:
    """Test relationship to transition mapping."""

    def test_causes_maps_to_causal(self):
        """Test CAUSES relationship maps to CAUSAL transition."""
        assert RELATIONSHIP_TO_TRANSITION[RelationshipType.CAUSES] == TransitionType.CAUSAL

    def test_contrasts_maps_to_adversative(self):
        """Test CONTRASTS relationship maps to ADVERSATIVE transition."""
        assert RELATIONSHIP_TO_TRANSITION[RelationshipType.CONTRASTS] == TransitionType.ADVERSATIVE

    def test_elaborates_maps_to_additive(self):
        """Test ELABORATES relationship maps to ADDITIVE transition."""
        assert RELATIONSHIP_TO_TRANSITION[RelationshipType.ELABORATES] == TransitionType.ADDITIVE

    def test_follows_maps_to_none(self):
        """Test FOLLOWS relationship maps to NONE transition."""
        assert RELATIONSHIP_TO_TRANSITION[RelationshipType.FOLLOWS] == TransitionType.NONE
