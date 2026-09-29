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
    from structure_score import paragraph_score
    s = paragraph_score(" ".join([A, B, C, D]), " ".join([A, B, C, D]))
    assert s["one_to_one"] == 1.0
    assert s["order"] == 1.0


def test_light_paraphrase_still_counts_as_one_to_one():
    from structure_score import paragraph_score
    out = ("A tiger crouches in long grass beside the river. Deer browse quietly in a clearing "
           "below the ridge. The wind carries the hunter's scent toward the herd. "
           "Startled animals scatter into the forest before an attack.")
    assert paragraph_score(" ".join([A, B, C, D]), out)["one_to_one"] == 1.0


def test_merged_sentences_are_not_one_to_one():
    from structure_score import paragraph_score
    out = (A[:-1] + ", while " + B[0].lower() + B[1:-1] + "; " + C[0].lower() + C[1:-1]
           + ", and " + D[0].lower() + D[1:])
    s = paragraph_score(" ".join([A, B, C, D]), out)
    assert s["one_to_one"] == 0.0
    assert s["n_out"] == 1


def test_reversed_order():
    from structure_score import paragraph_score
    s = paragraph_score(" ".join([A, B, C, D]), " ".join([D, C, B, A]))
    assert s["order"] == -1.0


def test_sentence_lengths():
    from structure_score import paragraph_score
    s = paragraph_score(" ".join([A, C]), " ".join([A, C]))
    assert s["mean_len"] == pytest.approx(10.0)
    assert s["sd_len"] == pytest.approx(0.0)


def test_document_pairs_paragraphs_and_skips_headings():
    from structure_score import document_score
    inp = f"# Title\n\n{A} {B}\n\n{C} {D}\n"
    out = f"# Title\n\n{A} {B}\n\n{D} {C}\n"
    s = document_score(inp, out)
    assert s["paragraphs"] == 2
    assert s["one_to_one"] == 1.0
    assert s["order"] == pytest.approx(0.0)  # one paragraph in order, one reversed


def test_document_paragraph_count_must_match():
    from structure_score import document_score
    with pytest.raises(ValueError):
        document_score(f"{A}\n\n{B}", f"{A} {B}")
