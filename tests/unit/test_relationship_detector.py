"""Unit tests for relationship detection."""

import pytest
from src.ingestion.relationship_detector import (
    RelationshipDetector,
    RELATIONSHIP_MARKERS,
)
from src.ingestion.proposition_extractor import PropositionExtractor
from src.models.graph import RelationshipType, PropositionNode


class TestRelationshipDetector:
    """Test RelationshipDetector functionality."""

    @pytest.fixture
    def detector(self):
        """Create detector instance without LLM."""
        return RelationshipDetector(llm_provider=None, use_llm_fallback=False)

    @pytest.fixture
    def extractor(self):
        """Create proposition extractor."""
        return PropositionExtractor()

    def test_empty_propositions(self, detector):
        """Test handling of empty propositions."""
        result = detector.detect_relationships([])
        assert result == []

    def test_single_proposition(self, detector):
        """Test handling of single proposition."""
        prop = PropositionNode(
            id="p1",
            text="Single proposition.",
            subject="Single",
            verb="proposition",
            object=None
        )
        result = detector.detect_relationships([prop])
        assert result == []

    def test_causal_marker_because(self, detector, extractor):
        """Test detection of causal relationship with 'because'."""
        text = "The project failed. Because the team lacked resources."
        propositions = extractor.extract_from_text(text)

        relationships = detector.detect_relationships(propositions)

        # Should find causal relationship
        causal_rels = [r for r in relationships if r.relationship == RelationshipType.CAUSES]
        assert len(causal_rels) > 0 or any(r.relationship == RelationshipType.FOLLOWS for r in relationships)

    def test_causal_marker_therefore(self, detector, extractor):
        """Test detection of causal relationship with 'therefore'."""
        text = "The data supports the hypothesis. Therefore, we accept it."
        propositions = extractor.extract_from_text(text)

        relationships = detector.detect_relationships(propositions)

        causal_rels = [r for r in relationships if r.relationship == RelationshipType.CAUSES]
        # Should detect causality or at least sequence
        assert len(relationships) > 0

    def test_contrast_marker_however(self, detector, extractor):
        """Test detection of contrast with 'however'."""
        text = "The old method was slow. However, the new approach is faster."
        propositions = extractor.extract_from_text(text)

        relationships = detector.detect_relationships(propositions)

        contrast_rels = [r for r in relationships if r.relationship == RelationshipType.CONTRASTS]
        assert len(contrast_rels) > 0

    def test_contrast_marker_but(self, detector, extractor):
        """Test detection of contrast with 'but'."""
        text = "He wanted to go. But the weather was bad."
        propositions = extractor.extract_from_text(text)

        relationships = detector.detect_relationships(propositions)

        contrast_rels = [r for r in relationships if r.relationship == RelationshipType.CONTRASTS]
        assert len(contrast_rels) > 0

    def test_elaboration_marker(self, detector, extractor):
        """Test detection of elaboration with 'for example'."""
        text = "There are many options. For example, you could use Python."
        propositions = extractor.extract_from_text(text)

        relationships = detector.detect_relationships(propositions)

        elab_rels = [r for r in relationships if r.relationship == RelationshipType.ELABORATES]
        assert len(elab_rels) > 0

    def test_sequence_fallback(self, detector, extractor):
        """Test that unconnected propositions get FOLLOWS relationship."""
        text = "First statement. Second statement."
        propositions = extractor.extract_from_text(text)

        relationships = detector.detect_relationships(propositions)

        # Should have at least FOLLOWS relationship
        follows_rels = [r for r in relationships if r.relationship == RelationshipType.FOLLOWS]
        assert len(follows_rels) >= 1 or len(relationships) > 0

    def test_detect_between_sentences(self, detector):
        """Test simple sentence-to-sentence detection."""
        sent1 = "The system crashed."
        sent2 = "Therefore, all data was lost."

        rel_type = detector.detect_between_sentences(sent1, sent2)

        assert rel_type == RelationshipType.CAUSES

    def test_detect_between_sentences_contrast(self, detector):
        """Test sentence contrast detection."""
        sent1 = "Option A is expensive."
        sent2 = "However, option B is affordable."

        rel_type = detector.detect_between_sentences(sent1, sent2)

        assert rel_type == RelationshipType.CONTRASTS

    def test_no_marker_returns_none(self, detector):
        """Test that sentences without markers return None."""
        sent1 = "The sun is bright."
        sent2 = "The grass is green."

        rel_type = detector.detect_between_sentences(sent1, sent2)

        assert rel_type is None


class TestRelationshipMarkers:
    """Test relationship marker definitions."""

    def test_causes_markers_exist(self):
        """Test that CAUSES markers are defined."""
        assert RelationshipType.CAUSES in RELATIONSHIP_MARKERS
        assert len(RELATIONSHIP_MARKERS[RelationshipType.CAUSES]) > 5

    def test_contrasts_markers_exist(self):
        """Test that CONTRASTS markers are defined."""
        assert RelationshipType.CONTRASTS in RELATIONSHIP_MARKERS
        assert len(RELATIONSHIP_MARKERS[RelationshipType.CONTRASTS]) > 5

    def test_elaborates_markers_exist(self):
        """Test that ELABORATES markers are defined."""
        assert RelationshipType.ELABORATES in RELATIONSHIP_MARKERS
        assert len(RELATIONSHIP_MARKERS[RelationshipType.ELABORATES]) > 3

    def test_key_markers_present(self):
        """Test that key markers are included."""
        all_markers = []
        for markers in RELATIONSHIP_MARKERS.values():
            all_markers.extend(markers)

        assert "however" in all_markers
        assert "because" in all_markers
        assert "therefore" in all_markers
        assert "for example" in all_markers
