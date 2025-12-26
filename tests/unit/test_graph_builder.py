"""Unit tests for semantic graph building."""

import pytest
from src.ingestion.graph_builder import SemanticGraphBuilder, DocumentGraphBuilder
from src.corpus.preprocessor import TextPreprocessor, ProcessedParagraph
from src.models.graph import (
    SemanticGraph,
    DocumentGraph,
    ParagraphRole,
    RhetoricalIntent,
)


class TestSemanticGraphBuilder:
    """Test SemanticGraphBuilder functionality."""

    @pytest.fixture
    def builder(self):
        """Create builder without LLM."""
        return SemanticGraphBuilder(llm_provider=None, use_llm_relationships=False)

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor."""
        return TextPreprocessor()

    def test_build_from_text_basic(self, builder):
        """Test building graph from basic text."""
        text = "The cat sat on the mat. It was comfortable."
        graph = builder.build_from_text(text)

        assert isinstance(graph, SemanticGraph)
        assert len(graph.nodes) >= 1
        assert graph.paragraph_idx == 0

    def test_build_from_text_with_role(self, builder):
        """Test building graph with specified role."""
        text = "In conclusion, this is the final point."
        graph = builder.build_from_text(text, paragraph_idx=5, role="CONCLUSION")

        assert graph.role == ParagraphRole.CONCLUSION
        assert graph.paragraph_idx == 5

    def test_build_from_paragraph(self, builder):
        """Test building graph from ProcessedParagraph."""
        para = ProcessedParagraph(
            text="The experiment proved the hypothesis. This was significant.",
            sentences=["The experiment proved the hypothesis.", "This was significant."],
            index=2,
            role="BODY"
        )

        graph = builder.build_from_paragraph(para)

        assert graph.paragraph_idx == 2
        assert graph.role == ParagraphRole.BODY
        assert len(graph.nodes) >= 1

    def test_graph_has_edges(self, builder):
        """Test that graph has relationship edges."""
        text = "The system failed. However, the backup worked."
        graph = builder.build_from_text(text)

        # Should have edges (at least FOLLOWS or CONTRASTS)
        assert len(graph.edges) >= 1

    def test_graph_summary(self, builder):
        """Test graph summary generation."""
        text = "Point one. Point two. Point three."
        graph = builder.build_from_text(text)

        summary = graph.to_summary()

        assert "Propositions:" in summary
        assert "P1" in summary

    def test_intent_detection_interrogative(self, builder):
        """Test detection of interrogative intent."""
        text = "What is the answer? Why does this happen? How can we solve it?"
        graph = builder.build_from_text(text)

        assert graph.intent == RhetoricalIntent.INTERROGATIVE

    def test_intent_detection_argument(self, builder):
        """Test default argument intent."""
        text = "This approach is better because it provides more value."
        graph = builder.build_from_text(text)

        assert graph.intent == RhetoricalIntent.ARGUMENT

    def test_validate_graph_valid(self, builder):
        """Test validation of valid graph."""
        text = "Valid sentence one. Valid sentence two."
        graph = builder.build_from_text(text)

        issues = builder.validate_graph(graph)

        assert len(issues) == 0

    def test_validate_graph_empty(self, builder):
        """Test validation of empty graph."""
        graph = SemanticGraph()

        issues = builder.validate_graph(graph)

        assert "no proposition nodes" in issues[0].lower()


class TestDocumentGraphBuilder:
    """Test DocumentGraphBuilder functionality."""

    @pytest.fixture
    def builder(self):
        """Create builder without LLM."""
        return DocumentGraphBuilder(llm_provider=None, use_llm_relationships=False)

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor."""
        return TextPreprocessor()

    def test_build_from_document(self, builder, preprocessor):
        """Test building document graph."""
        text = """This is the introduction paragraph.

        This is the body paragraph with more details. It explains the concept.

        This is the conclusion that wraps up."""

        document = preprocessor.process(text)
        doc_graph = builder.build_from_document(document)

        assert isinstance(doc_graph, DocumentGraph)
        assert len(doc_graph.paragraphs) == 3
        assert doc_graph.thesis  # Should have thesis
        assert doc_graph.intent  # Should have intent
        assert doc_graph.perspective  # Should have perspective

    def test_paragraph_roles_preserved(self, builder, preprocessor):
        """Test that paragraph roles are preserved."""
        text = """Introduction here.

        Body content here.

        Conclusion here."""

        document = preprocessor.process(text)
        doc_graph = builder.build_from_document(document)

        roles = [p.role for p in doc_graph.paragraphs]
        assert ParagraphRole.INTRO in roles
        assert ParagraphRole.CONCLUSION in roles

    def test_keywords_extracted(self, builder, preprocessor):
        """Test that keywords are extracted."""
        text = """The algorithm processes data efficiently.

        The algorithm uses advanced techniques for data processing.

        The algorithm achieves good results."""

        document = preprocessor.process(text)
        doc_graph = builder.build_from_document(document)

        # Should have keywords that appear across paragraphs
        assert len(doc_graph.keywords) >= 1

    def test_perspective_detection(self, builder, preprocessor):
        """Test perspective detection."""
        text = """I believe this is important.

        I think we should consider it.

        My conclusion is clear."""

        document = preprocessor.process(text)
        doc_graph = builder.build_from_document(document)

        assert doc_graph.perspective == "first_person_singular"

    def test_third_person_detection(self, builder, preprocessor):
        """Test third person perspective detection."""
        text = """The researcher conducted experiments.

        She analyzed the data carefully.

        The results showed significant findings."""

        document = preprocessor.process(text)
        doc_graph = builder.build_from_document(document)

        assert doc_graph.perspective == "third_person"

    def test_document_intent_persuade(self, builder, preprocessor):
        """Test detection of persuasive intent."""
        text = """We should adopt this approach.

        The evidence proves we must act now. I argue this is critical.

        Therefore, we should implement these changes."""

        document = preprocessor.process(text)
        doc_graph = builder.build_from_document(document)

        assert doc_graph.intent == "persuade"

    def test_document_intent_inform(self, builder, preprocessor):
        """Test detection of informative intent."""
        text = """This document describes the system.

        The system has several components.

        Each component serves a specific function."""

        document = preprocessor.process(text)
        doc_graph = builder.build_from_document(document)

        assert doc_graph.intent == "inform"
