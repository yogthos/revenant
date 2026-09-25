"""Tests for scripts/clean_corpus.py."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from clean_corpus import (  # noqa: E402
    clean_paragraph,
    fix_ocr,
    ocr_fix_acceptable,
    rebuild_ocr_book,
    replace_book,
)

BASE = "This sentence is plain filler so the paragraph clears the minimum length check easily. " * 3


class TestCleanParagraph:
    def test_strips_italic_and_bold_markup(self):
        out = clean_paragraph("If _everything_ were relative, draw =AB= and =O′= here. " + BASE)
        assert "If everything were relative, draw AB and O′ here." in out

    def test_double_hyphen_becomes_em_dash(self):
        out = clean_paragraph("My own belief--for which the reasons will appear--is simple. " + BASE)
        assert out.startswith("My own belief—for which the reasons will appear—is simple.")

    def test_trailing_colon_dash_becomes_colon(self):
        out = clean_paragraph(BASE + "The definitions are as follows:--")
        assert out.endswith("as follows:")

    def test_removes_footnote_markers(self):
        out = clean_paragraph("The eclipse was in 776 B. C.[1] No doubt[23] it was dark.* So** it went.[A] " + BASE)
        assert out.startswith("The eclipse was in 776 B. C. No doubt it was dark. So it went. ")

    def test_keeps_math_asterisk_free_text(self):
        out = clean_paragraph("Three times three is nine, and nine is a square. " + BASE)
        assert out.startswith("Three times three is nine")

    def test_removes_inline_citations(self):
        out = clean_paragraph('It is "of no physical importance" (4, p. 652). Later (p. 993) he agreed. ' + BASE)
        assert out.startswith('It is "of no physical importance". Later he agreed. ')

    def test_lowercases_all_caps_emphasis(self):
        out = clean_paragraph("There is one element which SEEMS common, and NEED NEVER HAVE BEEN SUPPRESSED. " + BASE)
        assert out.startswith("There is one element which seems common, and need never have been suppressed.")

    def test_keeps_roman_numerals_and_pronoun_i(self):
        out = clean_paragraph("In Lecture II I said that Chapter IV was wrong. " + BASE)
        assert out.startswith("In Lecture II I said that Chapter IV was wrong.")

    def test_strips_section_heading_prefix(self):
        out = clean_paragraph("(a) ACQUIRED HABITS.--In Lecture II we saw how animals learn. " + BASE)
        assert out.startswith("In Lecture II we saw how animals learn.")

    def test_strips_chapter_number_prefix(self):
        assert clean_paragraph("XVI Various periodic oscillations run through history. " + BASE).startswith("Various")
        assert clean_paragraph("Ill Modern life is built on science. " + BASE).startswith("Modern")
        assert clean_paragraph("I MEAN by an extrinsic law this. " + BASE).startswith("I mean by")

    @pytest.mark.parametrize("text", [
        "* There is a wide field of unconscious phenomena. " + BASE,
        "[12] See his book on the subject. " + BASE,
        "Let be a Hamiltonian coordinate and consider it. " + BASE,
        "(a) ; (b) is only zero when and are identical. " + BASE,
        "Precise, we must specify how much later [Math: e_{2}] is to occur. " + BASE,
    ])
    def test_drops_footnotes_formula_gaps_and_cut_paragraphs(self, text):
        assert clean_paragraph(text) is None

    @pytest.mark.parametrize("tail", ["Then its measured mass will be", "such a relation that--"])
    def test_trims_sentence_cut_off_by_a_formula(self, tail):
        assert clean_paragraph(BASE + tail) == BASE.strip()

    def test_drops_cut_paragraph_too_short_after_trimming(self):
        assert clean_paragraph("One full sentence here. Then its measured mass will be") is None

    def test_drops_short_paragraphs(self):
        assert clean_paragraph("Too short to keep.") is None


RAW_OCR = """INTRODUCTION: ON THE VALUE OF
SCEPTICISM

I wish to propose for the reader’s favourable consideration a doctrine which may appear wildly
paradoxical and subversive. It is a doctrine that a man is considered such a much greater man than a mere

INTRODUCTION
clerk, and is able to get so much more money — provided his beliefs are true. That is the con-
servative view, and there is little more to say about it that would be worth the reader's time today.

IDEALS OF HAPPINESS 103

Empire, while the doctrines of Confucius were eminently calculated to avoid friction between the
various classes of the population, and so they persisted for a very long time indeed in that country.

1 See The Mew Republic, February i , 192a, pp. 259 ff.

• Modified since the above was written.

1 68
Education is the next matter, and it is a matter on which a great deal could be said by anyone
who has the patience to consider it without prejudice, and without the passions of the moment.

III

Modern life is built on science in two respects, and each of them deserves a careful discussion here.
"""


class TestRebuildOcrBook:
    def test_rebuilds_paragraphs(self):
        paras = rebuild_ocr_book(RAW_OCR, min_words=10)
        assert len(paras) == 4
        first, empire, second, third = paras
        assert empire.startswith("Empire, while")
        assert "than a mere clerk, and is able" in first
        assert "the conservative view" in first
        assert "INTRODUCTION" not in first
        assert "IDEALS OF HAPPINESS" not in " ".join(paras)
        assert "Mew Republic" not in " ".join(paras)
        assert "Modified since" not in " ".join(paras)
        assert "1 68" not in second and second.startswith("Education is the next matter")
        assert third.startswith("Modern life")

    def test_page_break_continuation_with_capital_is_joined(self):
        paras = rebuild_ocr_book(RAW_OCR, min_words=10)
        # "...did not end a sentence, so the next page's text joins it"
        joined = [p for p in paras if "calculated to avoid friction" in p]
        assert not joined or "Empire, while" in joined[0]


class TestOcrFixGuard:
    def test_accepts_spelling_fixes(self):
        orig = "There remains, however, a residuu m which cannot be treated by sdence. T he desire s are strong."
        fixed = "There remains, however, a residuum which cannot be treated by science. The desires are strong."
        assert ocr_fix_acceptable(orig, fixed)

    def test_rejects_rewording(self):
        orig = "There remains, however, a residuum which cannot be treated by purely intellectual methods."
        fixed = "Still, some residue remains that pure intellect alone is unable to handle."
        assert not ocr_fix_acceptable(orig, fixed)

    def test_rejects_added_sentence(self):
        orig = "There remains, however, a residuum which cannot be treated by purely intellectual methods."
        fixed = orig + " This is an important point that deserves emphasis."
        assert not ocr_fix_acceptable(orig, fixed)


class TestFixOcr:
    def test_uses_cache_and_guard(self, tmp_path):
        cache = tmp_path / "cache.json"
        paras = ["The sdence of it is clear enough to everyone who looks.", "A plain sentence with nothing wrong."]
        call = MagicMock(side_effect=[
            "The science of it is clear enough to everyone who looks.",
            "An entirely different sentence the model made up on its own.",
        ])
        out = fix_ocr(paras, call, cache)
        assert out == ["The science of it is clear enough to everyone who looks.", paras[1]]
        # Second run reads the cache, no new calls.
        call.reset_mock()
        assert fix_ocr(paras, call, cache) == out
        call.assert_not_called()
        assert len(json.loads(cache.read_text())) == 2

    def test_prompt_asks_for_minimal_changes(self):
        from clean_corpus import OCR_SYSTEM_PROMPT
        prompt = OCR_SYSTEM_PROMPT.lower()
        assert "only" in prompt and "ocr" in prompt
        assert "do not" in prompt and "reword" in prompt


class TestReplaceBook:
    def test_replaces_contiguous_block(self):
        raw = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho"
        paras = ["Other book paragraph one.", "zz alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
                 "Other book paragraph two."]
        out = replace_book(paras, raw, ["New A.", "New B."])
        assert out == ["Other book paragraph one.", "New A.", "New B.", "Other book paragraph two."]


class TestLeftovers:
    def test_drops_paragraph_continuing_a_removed_formula(self):
        assert clean_paragraph("where is the frequency, and is Rydberg's constant as before. " + BASE) is None

    def test_removes_parenthetical_citations(self):
        out = clean_paragraph('Professor Watson ("Behavior," pp. 262-3) has an idea. Dr Jeans (Atomicity and Quanta, p. 8) says so. ' + BASE)
        assert out.startswith("Professor Watson has an idea. Dr Jeans says so.")

    def test_removes_lone_scanner_debris(self):
        out = clean_paragraph("There was bias. ^ It is just | in such matters ■ that it helps. " + BASE)
        assert out.startswith("There was bias. It is just in such matters that it helps.")
        assert clean_paragraph("^ There is, however, a case. " + BASE).startswith("There is, however")
        assert "connected with" in clean_paragraph("Events connected| with each other. " + BASE)

    def test_removes_stray_underscore(self):
        out = clean_paragraph("Namely, _All a priori knowledge deals with universals. " + BASE)
        assert out.startswith("Namely, All a priori")


RAW_SECTIONS = """The first essay ends here with a sentence that is complete and quite long enough to count as text.

i7 What would be the effect of a spread of rational scepticism among the people of the world at large?
The effect would be large, and I think it would be good, though it is hard to be certain of such things.

rv The tendency of culture in our time is, and will probably remain, towards a greater measure of
freedom in all the arts, and towards less respect for the rules of the past, which is all to the good. It is
the finest country on
1 See The Freeman, February 15, 1922, p. 532.
earth, and ought always to be enthusiastically supported in everything that it does by all good men.

The practical men who administered the empire were wise and understood what they were doing there.

a romantic admiration for action came before the war and it was common among the young of all classes.
"""


class TestRebuildLeftovers:
    def test_strips_ocr_section_numbers(self):
        paras = rebuild_ocr_book(RAW_SECTIONS, min_words=10)
        assert any(p.startswith("What would be the effect") for p in paras)
        assert any(p.startswith("The tendency of culture") for p in paras)

    def test_drops_footnote_lines_inside_block(self):
        paras = rebuild_ocr_book(RAW_SECTIONS, min_words=10)
        text = " ".join(paras)
        assert "The Freeman" not in text
        assert "the finest country on earth, and ought" in text

    def test_lowercase_block_joins_previous(self):
        paras = rebuild_ocr_book(RAW_SECTIONS, min_words=10)
        assert not any(p[0].islower() for p in paras)
        # Nothing ended mid-sentence, so it joins the previous paragraph.
        assert any("doing there. a romantic admiration" in p for p in paras)

    def test_lowercase_block_joins_last_unfinished_paragraph(self):
        raw = ("The vitalist is a temperamentally inactive man with a great deal to say about it, as always, and with\n\n"
               "> The chief argument against this tradition is that the book is not very long.\n\n"
               "a romantic admiration for action that is found in all men of his type and in many others too.\n")
        paras = rebuild_ocr_book(raw, min_words=10)
        assert any("and with a romantic admiration for action" in p for p in paras)
