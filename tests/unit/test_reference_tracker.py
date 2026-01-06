"""Tests for reference tracker module."""

import pytest
from src.validation.reference_tracker import (
    extract_references,
    reinject_references,
    strip_references,
    Reference,
    ReferenceMap,
    _extract_attached_word,
    _find_injection_point,
    _find_fuzzy_injection_point,
)


class TestExtractReferences:
    """Tests for extract_references function."""

    def test_footnote_references(self):
        """Test extraction of footnote references [^N]."""
        text = "Einstein[^1] developed relativity[^2] in 1905."
        clean, ref_map = extract_references(text)

        assert clean == "Einstein developed relativity in 1905."
        assert len(ref_map.references) == 2
        assert ref_map.references[0].marker == "[^1]"
        assert ref_map.references[0].attached_to == "Einstein"
        assert ref_map.references[1].marker == "[^2]"
        assert ref_map.references[1].attached_to == "relativity"

    def test_citation_references(self):
        """Test extraction of citation references [N]."""
        text = "Previous work[1] shows results[2,3] for models."
        clean, ref_map = extract_references(text)

        assert clean == "Previous work shows results for models."
        assert len(ref_map.references) == 2
        assert ref_map.references[0].marker == "[1]"
        assert ref_map.references[1].marker == "[2,3]"

    def test_named_footnotes(self):
        """Test extraction of named footnotes [^name]."""
        text = "See the documentation[^docs] for details[^note]."
        clean, ref_map = extract_references(text)

        assert clean == "See the documentation for details."
        assert len(ref_map.references) == 2
        assert ref_map.references[0].ref_id == "docs"
        assert ref_map.references[1].ref_id == "note"

    def test_no_references(self):
        """Test text without references."""
        text = "This is plain text without any references."
        clean, ref_map = extract_references(text)

        assert clean == text
        assert not ref_map.has_references()
        assert len(ref_map.references) == 0

    def test_multiple_spaces_cleaned(self):
        """Test that double spaces are cleaned after extraction."""
        text = "Word[^1]  with  spaces[^2]  here."
        clean, ref_map = extract_references(text)

        assert "  " not in clean

    def test_proper_noun_attachment(self):
        """Test reference attached to multi-word proper nouns."""
        text = "Karl Marx[^1] wrote about capitalism."
        clean, ref_map = extract_references(text)

        assert ref_map.references[0].attached_to == "Karl Marx"

    def test_quoted_text_attachment(self):
        """Test reference attached to quoted text."""
        text = 'The phrase "cogito ergo sum"[^1] is famous.'
        clean, ref_map = extract_references(text)

        assert ref_map.references[0].attached_to == "cogito ergo sum"

    def test_parenthetical_attachment(self):
        """Test reference attached to parenthetical."""
        text = "The theory (special relativity)[^1] was revolutionary."
        clean, ref_map = extract_references(text)

        assert ref_map.references[0].attached_to == "special relativity"


class TestReinjectReferences:
    """Tests for reinject_references function."""

    def test_basic_reinjection(self):
        """Test basic reference reinjection."""
        text = "Einstein[^1] developed relativity[^2]."
        clean, ref_map = extract_references(text)

        styled = "Einstein developed relativity."
        result = reinject_references(styled, ref_map)

        assert "[^1]" in result
        assert "[^2]" in result
        assert "Einstein[^1]" in result
        assert "relativity[^2]" in result

    def test_reinjection_with_style_changes(self):
        """Test reinjection when text has been restyled."""
        text = "Einstein[^1] developed relativity[^2] in 1905."
        clean, ref_map = extract_references(text)

        styled = "The physicist Einstein formulated his theory of relativity in 1905."
        result = reinject_references(styled, ref_map)

        assert "Einstein[^1]" in result
        assert "relativity[^2]" in result

    def test_reinjection_empty_map(self):
        """Test reinjection with no references."""
        ref_map = ReferenceMap()
        styled = "Some styled text."
        result = reinject_references(styled, ref_map)

        assert result == styled

    def test_reinjection_missing_entity(self):
        """Test reinjection when entity is completely removed."""
        text = "Einstein[^1] was a physicist."
        clean, ref_map = extract_references(text)

        # Entity completely missing from output
        styled = "A great physicist changed physics."
        result = reinject_references(styled, ref_map)

        # Reference should not appear (no injection point found)
        assert result == styled

    def test_context_disambiguation(self):
        """Test that context helps disambiguate multiple matches."""
        text = "Einstein[^1] was born in Germany. Einstein later moved to America."
        clean, ref_map = extract_references(text)

        styled = "Einstein was born in Germany. Einstein later moved to America."
        result = reinject_references(styled, ref_map)

        # Should inject after first Einstein (context: "Germany")
        assert result.count("[^1]") == 1
        # Reference should be near "Germany" context
        idx_ref = result.find("[^1]")
        idx_germany = result.find("Germany")
        assert idx_ref < idx_germany  # Reference before Germany mention

    def test_fuzzy_matching_last_name(self):
        """Test fuzzy matching with last name only."""
        text = "Albert Einstein[^1] was brilliant."
        clean, ref_map = extract_references(text)

        # Output only uses last name
        styled = "Einstein was a brilliant physicist."
        result = reinject_references(styled, ref_map)

        assert "Einstein[^1]" in result


class TestExtractAttachedWord:
    """Tests for _extract_attached_word helper."""

    def test_simple_word(self):
        """Test extraction of simple word."""
        result = _extract_attached_word("The physicist Einstein")
        assert result == "Einstein"

    def test_proper_noun_pair(self):
        """Test extraction of two-word proper noun."""
        result = _extract_attached_word("The author Karl Marx")
        assert result == "Karl Marx"

    def test_quoted_text(self):
        """Test extraction from quoted text."""
        result = _extract_attached_word('He said "hello world"')
        assert result == "hello world"

    def test_parenthetical(self):
        """Test extraction from parenthetical."""
        result = _extract_attached_word("The theory (general relativity)")
        assert result == "general relativity"

    def test_with_punctuation(self):
        """Test stripping punctuation from word."""
        result = _extract_attached_word("Einstein,")
        assert result == "Einstein"

    def test_empty_prefix(self):
        """Test with empty prefix."""
        result = _extract_attached_word("")
        assert result == ""


class TestFindInjectionPoint:
    """Tests for _find_injection_point helper."""

    def test_single_match(self):
        """Test finding injection point with single match."""
        text = "The physicist Einstein was brilliant."
        point = _find_injection_point(text, "Einstein", "")

        assert point is not None
        assert text[:point].endswith("Einstein")

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        text = "The physicist EINSTEIN was brilliant."
        point = _find_injection_point(text, "einstein", "")

        assert point is not None

    def test_no_match(self):
        """Test when word not found."""
        text = "The physicist was brilliant."
        point = _find_injection_point(text, "Einstein", "")

        assert point is None

    def test_word_boundary(self):
        """Test that partial matches are rejected."""
        text = "The Einsteinian theory was novel."
        point = _find_injection_point(text, "Einstein", "")

        # Should not match "Einsteinian"
        assert point is None


class TestFindFuzzyInjectionPoint:
    """Tests for _find_fuzzy_injection_point helper."""

    def test_last_name_match(self):
        """Test matching by last name only."""
        text = "Einstein was a great physicist."
        point = _find_fuzzy_injection_point(text, "Albert Einstein")

        assert point is not None
        assert text[:point].endswith("Einstein")

    def test_short_word_rejected(self):
        """Test that very short words are rejected."""
        text = "The cat sat on the mat."
        point = _find_fuzzy_injection_point(text, "at")

        assert point is None


class TestStripReferences:
    """Tests for strip_references function."""

    def test_strip_all_references(self):
        """Test stripping all reference types."""
        text = "Einstein[^1] and Bohr[2] debated[^note] physics."
        result = strip_references(text)

        assert "[^1]" not in result
        assert "[2]" not in result
        assert "[^note]" not in result
        assert "Einstein" in result
        assert "Bohr" in result

    def test_no_double_spaces(self):
        """Test that double spaces are cleaned."""
        text = "Word[^1]  more[^2]  text."
        result = strip_references(text)

        assert "  " not in result


class TestReferenceMap:
    """Tests for ReferenceMap class."""

    def test_add_reference(self):
        """Test adding references to map."""
        ref_map = ReferenceMap()
        ref = Reference(
            marker="[^1]",
            ref_id="1",
            attached_to="Einstein",
            context="Einstein developed...",
            position=8,
        )
        ref_map.add(ref)

        assert ref_map.has_references()
        assert len(ref_map.references) == 1

    def test_get_refs_for_entity(self):
        """Test retrieving references by entity."""
        ref_map = ReferenceMap()
        ref = Reference(
            marker="[^1]",
            ref_id="1",
            attached_to="Einstein",
            context="",
            position=0,
        )
        ref_map.add(ref)

        # Case-insensitive lookup
        refs = ref_map.get_refs_for_entity("einstein")
        assert len(refs) == 1
        assert refs[0].marker == "[^1]"

    def test_multiple_refs_same_entity(self):
        """Test multiple references to same entity."""
        ref_map = ReferenceMap()
        ref_map.add(Reference("[^1]", "1", "Einstein", "", 0))
        ref_map.add(Reference("[^2]", "2", "Einstein", "", 50))

        refs = ref_map.get_refs_for_entity("Einstein")
        assert len(refs) == 2
