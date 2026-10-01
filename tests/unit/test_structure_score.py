"""structure_score measures how much of the input's sentence structure the output kept."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

A = "The tiger crouches in the long grass beside the river."
B = "Deer browse quietly in the clearing below the ridge."
C = "Wind carries the scent of the hunter toward the herd."
D = "Startled animals scatter into the forest before any attack."


def test_identical_paragraph_is_all_one_to_one():
    from src.utils.structure import paragraph_score
    s = paragraph_score(" ".join([A, B, C, D]), " ".join([A, B, C, D]))
    assert s["one_to_one"] == 1.0
    assert s["order"] == 1.0


def test_light_paraphrase_still_counts_as_one_to_one():
    from src.utils.structure import paragraph_score
    out = ("A tiger crouches in long grass beside the river. Deer browse quietly in a clearing "
           "below the ridge. The wind carries the hunter's scent toward the herd. "
           "Startled animals scatter into the forest before an attack.")
    assert paragraph_score(" ".join([A, B, C, D]), out)["one_to_one"] == 1.0


def test_merged_sentences_are_not_one_to_one():
    from src.utils.structure import paragraph_score
    out = (A[:-1] + ", while " + B[0].lower() + B[1:-1] + "; " + C[0].lower() + C[1:-1]
           + ", and " + D[0].lower() + D[1:])
    s = paragraph_score(" ".join([A, B, C, D]), out)
    assert s["one_to_one"] == 0.0
    assert s["n_out"] == 1


def test_reversed_order():
    from src.utils.structure import paragraph_score
    s = paragraph_score(" ".join([A, B, C, D]), " ".join([D, C, B, A]))
    assert s["order"] == -1.0


def test_sentence_lengths():
    from src.utils.structure import paragraph_score
    s = paragraph_score(" ".join([A, C]), " ".join([A, C]))
    assert s["mean_len"] == pytest.approx(10.0)
    assert s["sd_len"] == pytest.approx(0.0)


def test_document_pairs_paragraphs_and_skips_headings():
    from src.utils.structure import document_score
    inp = f"# Title\n\n{A} {B}\n\n{C} {D}\n"
    out = f"# Title\n\n{A} {B}\n\n{D} {C}\n"
    s = document_score(inp, out)
    assert s["paragraphs"] == 2
    assert s["one_to_one"] == 1.0
    assert s["order"] == pytest.approx(0.0)  # one paragraph in order, one reversed


def test_document_paragraph_count_must_match():
    from src.utils.structure import document_score
    with pytest.raises(ValueError):
        document_score(f"{A}\n\n{B}", f"{A} {B}")


class TestShuffleSentences:
    def _text(self, n):
        return " ".join(f"Sentence number {w} talks about topic {w} at length." for w in
                        ["one", "two", "three", "four", "five", "six", "seven"][:n])

    def test_keeps_every_sentence_in_a_new_order(self):
        import random
        from src.utils.nlp import split_into_sentences
        from src.utils.structure import kendall_tau, shuffle_sentences
        text = self._text(6)
        out = shuffle_sentences(text, random.Random(1))
        before, after = split_into_sentences(text), split_into_sentences(out)
        assert sorted(before) == sorted(after)
        assert kendall_tau([before.index(s) for s in after]) <= 0.2

    def test_two_sentences_are_swapped(self):
        import random
        from src.utils.structure import shuffle_sentences
        out = shuffle_sentences(f"{A} {B}", random.Random(0))
        assert out == f"{B} {A}"

    def test_one_sentence_is_unchanged(self):
        import random
        from src.utils.structure import shuffle_sentences
        assert shuffle_sentences(A, random.Random(0)) == A

    def test_same_seed_same_order(self):
        import random
        from src.utils.structure import shuffle_sentences
        text = self._text(7)
        assert shuffle_sentences(text, random.Random(5)) == shuffle_sentences(text, random.Random(5))


class TestCopiedRuns:
    CORPUS = ("The man who has no tincture of philosophy goes through life imprisoned in the prejudices "
              "derived from common sense, from the habitual beliefs of his age or his nation.")

    def test_finds_the_longest_run_shared_with_the_corpus(self):
        from src.utils.structure import longest_copied_run, ngram_index
        index = ngram_index(self.CORPUS, n=4)
        text = ("Today I read that the man who has no tincture of philosophy goes through life "
                "quite happily.")
        assert longest_copied_run(text, index, n=4) == 11  # "the man ... through life"

    def test_ignores_case_and_punctuation(self):
        from src.utils.structure import longest_copied_run, ngram_index
        index = ngram_index(self.CORPUS, n=4)
        assert longest_copied_run("From Common Sense; from the habitual beliefs!", index, n=4) == 7

    def test_no_shared_ngram_is_zero(self):
        from src.utils.structure import longest_copied_run, ngram_index
        index = ngram_index(self.CORPUS, n=4)
        assert longest_copied_run("Stocks rose sharply in May on chip demand.", index, n=4) == 0

    def test_document_score_reports_the_longest_copy(self):
        from src.utils.structure import document_score, ngram_index
        index = ngram_index(self.CORPUS, n=4)
        out = f"{A} The man who has no tincture of philosophy goes on. {B}"
        s = document_score(f"{A} {B}", out, corpus_index=index, n=4)
        assert s["copied"] == 9  # "the man ... philosophy goes"


class TestOpenerKept:
    """A paragraph's opening sentence often carries the transition from the one before."""

    def test_opener_kept_when_the_output_starts_from_the_input_opener(self):
        from src.utils.structure import paragraph_score
        out = "A tiger crouches beside the river in long grass. " + " ".join([C, B, D])
        assert paragraph_score(" ".join([A, B, C, D]), out)["opener_kept"] is True

    def test_opener_lost_when_it_moves_into_the_middle(self):
        from src.utils.structure import paragraph_score
        assert paragraph_score(" ".join([A, B, C, D]), " ".join([C, A, B, D]))["opener_kept"] is False

    def test_document_reports_the_share_of_paragraphs(self):
        from src.utils.structure import document_score
        inp = f"{A} {B}\n\n{C} {D}"
        out = f"{A} {B}\n\n{D} {C}"
        assert document_score(inp, out)["opener_kept"] == 0.5
