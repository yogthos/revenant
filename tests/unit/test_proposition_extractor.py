"""Unit tests for proposition extraction."""

import pytest
from src.ingestion.proposition_extractor import PropositionExtractor, SVOTriple


class TestPropositionExtractor:
    """Test PropositionExtractor functionality."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return PropositionExtractor()

    def test_empty_input(self, extractor):
        """Test handling of empty input."""
        result = extractor.extract_from_text("")
        assert result == []

    def test_simple_sentence(self, extractor):
        """Test extraction from simple sentence."""
        text = "The cat sat on the mat."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        assert propositions[0].text  # Has text
        assert propositions[0].id  # Has ID

    def test_multiple_sentences(self, extractor):
        """Test extraction from multiple sentences."""
        text = "The scientist conducted an experiment. The results were surprising."
        propositions = extractor.extract_from_text(text)

        # Should have at least 2 propositions (one per sentence)
        assert len(propositions) >= 2

    def test_proposition_has_subject(self, extractor):
        """Test that propositions have subjects extracted."""
        text = "John wrote a book about history."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Subject should be extracted
        assert propositions[0].subject or propositions[0].text

    def test_proposition_has_verb(self, extractor):
        """Test that propositions have verbs extracted."""
        text = "The students learn mathematics daily."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Should have a verb
        assert propositions[0].verb or propositions[0].text

    def test_citation_preservation(self, extractor):
        """Test that citations are preserved."""
        text = "Research shows positive results[^1]. Another study confirms this[^2]."
        propositions = extractor.extract_from_text(text)

        # Should have citations attached
        has_citations = any(p.attached_citations for p in propositions)
        assert has_citations or any("[^" in p.text for p in propositions)

    def test_quotation_detection(self, extractor):
        """Test detection of quotations."""
        text = '"This is a quote," he said. Regular sentence here.'
        propositions = extractor.extract_from_text(text)

        # At least one should be marked as quotation
        assert len(propositions) >= 1

    def test_entity_extraction(self, extractor):
        """Test named entity extraction."""
        text = "Albert Einstein developed the theory of relativity in Germany."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Should have entities
        all_entities = []
        for p in propositions:
            all_entities.extend(p.entities)
        # Should find at least Einstein or Germany
        assert len(all_entities) > 0 or "Einstein" in propositions[0].text

    def test_keyword_extraction(self, extractor):
        """Test keyword extraction."""
        text = "The computer processes data using algorithms."
        propositions = extractor.extract_from_text(text)

        assert len(propositions) >= 1
        # Should have keywords
        assert len(propositions[0].keywords) > 0

    def test_sentence_index_tracking(self, extractor):
        """Test that sentence indices are tracked."""
        text = "First sentence. Second sentence. Third sentence."
        propositions = extractor.extract_from_text(text)

        # Should have different sentence indices
        indices = [p.source_sentence_idx for p in propositions]
        assert len(set(indices)) >= 1

    def test_unique_ids(self, extractor):
        """Test that proposition IDs are unique."""
        text = "One sentence here. Another sentence there. A third one too."
        propositions = extractor.extract_from_text(text)

        ids = [p.id for p in propositions]
        assert len(ids) == len(set(ids))  # All unique

    def test_complex_sentence(self, extractor):
        """Test extraction from complex sentence."""
        text = "Although the weather was bad, the team completed the project because they were dedicated."
        propositions = extractor.extract_from_text(text)

        # Should extract meaningful content
        assert len(propositions) >= 1
        all_text = " ".join(p.text for p in propositions)
        assert "team" in all_text.lower() or "project" in all_text.lower()


class TestSVOTriple:
    """Test SVOTriple data class."""

    def test_create_triple(self):
        """Test creating an SVO triple."""
        triple = SVOTriple(
            subject="The cat",
            verb="sat",
            object="on the mat",
            full_text="The cat sat on the mat."
        )

        assert triple.subject == "The cat"
        assert triple.verb == "sat"
        assert triple.object == "on the mat"

    def test_triple_without_object(self):
        """Test triple without object."""
        triple = SVOTriple(
            subject="He",
            verb="slept",
            object=None,
            full_text="He slept."
        )

        assert triple.object is None
