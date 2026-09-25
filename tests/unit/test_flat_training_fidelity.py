"""Fidelity fixes in scripts/generate_flat_training.py and scripts/filter_training_data.py."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# strip_modifiers
# ---------------------------------------------------------------------------

@pytest.mark.spacy_dependent
class TestStripModifiersKeepsMeaning:
    @pytest.mark.parametrize("word", ["never", "only", "always", "not", "rarely", "almost"])
    def test_keeps_meaning_adverbs(self, word):
        from generate_flat_training import strip_modifiers
        text = f"Philosophers {word} agree about the nature of truth and knowledge."
        assert word in strip_modifiers(text).lower().split()

    def test_keeps_predicative_adjectives(self):
        from generate_flat_training import strip_modifiers
        out = strip_modifiers("The problem is hard and the answer is simple.")
        assert "hard" in out and "simple" in out

    def test_keeps_quantity_and_comparative_adjectives(self):
        from generate_flat_training import strip_modifiers
        out = strip_modifiers("Many older men made fewer mistakes than the best students.")
        for word in ("Many", "older", "fewer", "best"):
            assert word in out

    def test_drops_plain_attributive_adjectives_and_manner_adverbs(self):
        from generate_flat_training import strip_modifiers
        out = strip_modifiers("The clever student quickly solved the difficult problem.")
        assert "clever" not in out
        assert "quickly" not in out
        assert "student" in out and "problem" in out


# ---------------------------------------------------------------------------
# Lexical bleed
# ---------------------------------------------------------------------------

class TestLexicalBleedTokenization:
    def test_punctuation_does_not_hide_shared_words(self):
        from generate_flat_training import check_lexical_bleed
        styled = "Superstition, cruelty, dogma."
        neutral = "superstition cruelty dogma"
        ok, ratio = check_lexical_bleed(neutral, styled)
        assert ratio == 1.0
        assert not ok

    def test_distinct_vocabulary_passes(self):
        from generate_flat_training import check_lexical_bleed
        ok, ratio = check_lexical_bleed("People fear new ideas.", "Men dread novelty; innovation terrifies.")
        assert ok and ratio == 0.0


# ---------------------------------------------------------------------------
# Perspective variants
# ---------------------------------------------------------------------------

class TestPerspectiveVariants:
    ORIGINAL = "I went to the lecture. I thought the speaker was wrong about everything he said."

    def test_identical_variant_rejected(self):
        from generate_flat_training import validate_perspective_variation
        ok, reason = validate_perspective_variation(self.ORIGINAL, self.ORIGINAL)
        assert not ok

    def test_near_identical_variant_rejected(self):
        from generate_flat_training import validate_perspective_variation
        varied = self.ORIGINAL.replace("he said", "he says")
        ok, _ = validate_perspective_variation(self.ORIGINAL, varied)
        assert not ok

    def test_real_change_accepted(self):
        from generate_flat_training import validate_perspective_variation
        varied = "We went to the lecture. We thought the speaker was wrong about everything he said."
        ok, reason = validate_perspective_variation(self.ORIGINAL, varied)
        assert ok, reason

    def test_skips_text_without_first_person_singular(self):
        import generate_flat_training as gft
        with patch.object(gft, "call_deepseek") as call:
            out = gft.create_perspective_variation(
                "Philosophy begins with doubt. Men fear thought.", "Bertrand Russell", "third_person"
            )
        assert out is None
        call.assert_not_called()


# ---------------------------------------------------------------------------
# Snowflake (topic swap) variants
# ---------------------------------------------------------------------------

class TestSnowflakeKeepsStructure:
    ORIGINAL = (
        "Religion, as I understand it, is based primarily upon fear. "
        "It is partly the terror of the unknown; it is partly the wish to feel that you have a big brother. "
        "Fear is the basis of the whole thing."
    )
    SWAPPED = (
        "Gardening, as I understand it, is based primarily upon patience. "
        "It is partly the waiting for the rain; it is partly the wish to feel that you have a green thumb. "
        "Patience is the basis of the whole thing."
    )

    def test_accepts_topic_swap_with_same_structure(self):
        from generate_flat_training import validate_variation
        ok, reason = validate_variation(self.ORIGINAL, self.SWAPPED)
        assert ok, reason

    def test_rejects_unchanged_text(self):
        from generate_flat_training import validate_variation
        ok, _ = validate_variation(self.ORIGINAL, self.ORIGINAL)
        assert not ok

    def test_rejects_free_rewrite(self):
        from generate_flat_training import validate_variation
        rewrite = (
            "Gardening teaches patience above everything else in life, and a gardener soon learns this. "
            "You wait for rain, you wait for sun, and you hope the seeds were good ones to begin with. "
            "Nothing about it can be hurried."
        )
        ok, _ = validate_variation(self.ORIGINAL, rewrite)
        assert not ok

    def test_rejects_changed_punctuation_pattern(self):
        from generate_flat_training import validate_variation
        changed = self.SWAPPED.replace(", as I understand it,", " as I understand it").replace(";", ",")
        ok, _ = validate_variation(self.ORIGINAL, changed)
        assert not ok

    def test_prompt_asks_to_keep_wording(self):
        import generate_flat_training as gft
        with patch.object(gft, "call_deepseek", return_value=self.SWAPPED) as call:
            out = gft.create_topic_variation(self.ORIGINAL, "Bertrand Russell", "gardening")
        assert out == self.SWAPPED
        prompt = call.call_args.args[0] + call.call_args.args[1]
        assert "only the words tied to the topic" in prompt.lower()
        assert "keep every other word" in prompt.lower()


# ---------------------------------------------------------------------------
# Content label comes from the input, like at inference
# ---------------------------------------------------------------------------

class TestContentLabelFromInput:
    def test_classifier_sees_neutral_input(self):
        import generate_flat_training as gft
        from src.utils.content_classifier import ContentType
        with patch.object(gft, "classify_content_type", return_value=ContentType.CONCEPTUAL) as classify:
            gft.format_training_example(
                neutral_text="NEUTRAL INPUT TEXT here.",
                styled_text="STYLED OUTPUT TEXT here.",
                author="Bertrand Russell",
                word_count=4,
                output_format="mlx",
            )
        assert classify.call_args.args[0] == "NEUTRAL INPUT TEXT here."


# ---------------------------------------------------------------------------
# Source paragraph ids
# ---------------------------------------------------------------------------

class TestSourceIds:
    def test_expand_tags_every_item_with_its_paragraph(self):
        import generate_flat_training as gft
        paragraphs = ["First para text.", "Second para text."]
        items = gft.expand_corpus_with_variations(paragraphs, "X", skip_variation=True)
        assert items == [("First para text.", "original", (0,)), ("Second para text.", "original", (1,))]

    def test_overlapping_chunks_record_spanned_paragraphs(self):
        import generate_flat_training as gft
        sentence = "This sentence has exactly eight words in it. "
        paras = [(sentence * 5, "original", (i,)) for i in range(4)]
        paras.append(("Snow " * 60, "snowflake", (2,)))
        config = gft.OverlapConfig(min_words=60, max_words=100, overlap_sentences=1)
        chunks = gft.create_overlapping_chunks(paras, config)
        originals = [c for c in chunks if c[1] == "original"]
        assert any(len(src) > 1 for _, _, src in originals)
        assert all(set(src) <= {0, 1, 2, 3} for _, _, src in originals)
        assert ("Snow " * 60, "snowflake", (2,)) in chunks

    def test_intermediate_round_trip_keeps_ids(self, tmp_path):
        import generate_flat_training as gft
        items = [("a b c", "original", (0, 1)), ("d e f", "snowflake", (3,))]
        path = tmp_path / "items.json"
        gft.save_intermediate(items, path)
        assert gft.load_intermediate(path) == items

    def test_old_intermediate_without_ids_loads(self, tmp_path):
        import generate_flat_training as gft
        path = tmp_path / "old.json"
        path.write_text(json.dumps([{"text": "a b", "variation_type": "original"}]))
        assert gft.load_intermediate(path) == [("a b", "original", ())]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume:
    @pytest.fixture
    def fake_rtt(self):
        import generate_flat_training as gft
        neutralizer = MagicMock(spec=["neutralize"])
        neutralizer.neutralize.side_effect = lambda text, **kw: "plain words only " + str(len(text))
        with patch.object(gft, "get_rtt_neutralizer", return_value=(neutralizer, MagicMock())), \
             patch.object(gft, "create_input_variants", side_effect=lambda styled, neutral: [(neutral, "standard")]), \
             patch.object(gft, "check_lexical_bleed", return_value=(True, 0.0)):
            yield neutralizer

    CHUNKS = [
        ("Styled text number one with enough words.", "original", (0,)),
        ("Styled text number two with enough words.", "original", (1,)),
        ("Styled text number three with enough words.", "original", (2,)),
    ]

    def test_llama_factory_rows_carry_source_ids(self, tmp_path, fake_rtt):
        import generate_flat_training as gft
        out = tmp_path / "train.jsonl"
        gft.generate_training_data(self.CHUNKS, "X", out, output_format="llama_factory")
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["source_idx"] for r in rows] == [0, 1, 2]
        assert [r["source_paragraphs"] for r in rows] == [[0], [1], [2]]

    def test_resume_keeps_llama_factory_rows(self, tmp_path, fake_rtt):
        import generate_flat_training as gft
        out = tmp_path / "train.jsonl"
        gft.generate_training_data(self.CHUNKS[:2], "X", out, output_format="llama_factory")
        gft.generate_training_data(self.CHUNKS, "X", out, resume=True, output_format="llama_factory")
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["source_idx"] for r in rows] == [0, 1, 2]

    def test_resume_refuses_to_truncate_rows_without_ids(self, tmp_path, fake_rtt):
        import generate_flat_training as gft
        out = tmp_path / "train.jsonl"
        out.write_text(json.dumps({"instruction": "i", "input": "x", "output": "y"}) + "\n")
        with pytest.raises(SystemExit):
            gft.generate_training_data(self.CHUNKS, "X", out, resume=True, output_format="llama_factory")
        assert out.read_text().count("\n") == 1


# ---------------------------------------------------------------------------
# Finalize: filtering, grouped split, dataset_info
# ---------------------------------------------------------------------------

LONG_IN = "People are afraid of new ideas, and they are more afraid of them than of ruin or of death."
LONG_OUT = "Men fear thought as they fear nothing else on earth, more than ruin, more even than death."


def _row(src, output=LONG_OUT, inp=LONG_IN, idx=0):
    return {"instruction": "Do it.", "input": inp, "output": output,
            "source_idx": idx, "source_paragraphs": list(src)}


class TestGroupedSplit:
    def test_no_source_paragraph_in_both_splits(self):
        from filter_training_data import split_by_source
        rows = [_row((i, i + 1), idx=i) for i in range(200)] + [_row((i,), idx=500 + i) for i in range(200)]
        train, val, dropped = split_by_source(rows, val_fraction=0.1, seed=1, block_size=10)
        train_src = {p for r in train for p in r["source_paragraphs"]}
        val_src = {p for r in val for p in r["source_paragraphs"]}
        assert val and train
        assert not train_src & val_src
        assert len(train) + len(val) + dropped == len(rows)

    def test_rows_without_ids_grouped_by_output(self):
        from filter_training_data import split_by_source
        rows = [{"instruction": "i", "input": f"in {i}", "output": f"out {i // 3}"} for i in range(300)]
        train, val, _ = split_by_source(rows, val_fraction=0.1, seed=1)
        assert not {r["output"] for r in train} & {r["output"] for r in val}


class TestRowChecks:
    def test_rejects_placeholder_residue(self):
        from filter_training_data import row_problem
        assert row_problem(_row((0,), inp="The idea __ENT3__ held was odd, and people argued about it for many long years after.")) is not None

    def test_rejects_leading_punctuation(self):
        from filter_training_data import row_problem
        assert row_problem(_row((0,), inp=". " + LONG_IN)) is not None

    def test_rejects_lexical_bleed(self):
        from filter_training_data import row_problem
        row = _row((0,), output="Superstition, cruelty, dogma.", inp="superstition cruelty dogma " * 6)
        assert row_problem(row) is not None

    def test_accepts_clean_row(self):
        from filter_training_data import row_problem
        assert row_problem(_row((0,))) is None


class TestTwoWayEntailment:
    class FakeNLI:
        """Entails when every word of the hypothesis appears in the premise."""
        def predict(self, pairs, **kwargs):
            import numpy as np
            out = []
            for premise, hyp in pairs:
                words = {w.strip(".,").lower() for w in hyp.split()}
                prem = {w.strip(".,").lower() for w in premise.split()}
                entailed = words <= prem
                # label order: contradiction, entailment, neutral
                out.append([0.0, 5.0, 0.0] if entailed else [0.0, -5.0, 5.0])
            return np.array(out)

    class ParagraphConfusedNLI(FakeNLI):
        """Like nli-deberta-v3-small: long multi-sentence premises confuse it."""
        def predict(self, pairs, **kwargs):
            import numpy as np
            rows = super().predict(pairs, **kwargs)
            for i, (premise, _) in enumerate(pairs):
                if premise.count(". ") >= 3:
                    rows[i] = [0.0, -5.0, 5.0]
            return np.array(rows)

    def test_checks_sentences_against_short_aligned_spans(self):
        # A paragraph must entail itself even when the model can't read the
        # whole paragraph as one premise.
        from filter_training_data import entailment_problem
        text = ("The problem of liberty does not arise among savages. It arises among civilized men. "
                "Government grows as they grow. Freedom becomes more urgent. Nobody escapes the question.")
        assert entailment_problem(text, text, self.ParagraphConfusedNLI()) is None

    def test_rejects_added_content(self):
        from filter_training_data import entailment_problem
        neutral = "Men fear new ideas. They fear thought."
        styled = "Men fear new ideas. They fear thought. Also the moon is made of cheese."
        assert entailment_problem(neutral, styled, self.FakeNLI(), min_fraction=0.9) is not None

    def test_rejects_dropped_content(self):
        from filter_training_data import entailment_problem
        neutral = "Men fear new ideas. They fear thought. The moon is made of cheese."
        styled = "Men fear new ideas. They fear thought."
        assert entailment_problem(neutral, styled, self.FakeNLI(), min_fraction=0.9) is not None

    def test_accepts_equivalent_pair(self):
        from filter_training_data import entailment_problem
        text = "Men fear new ideas. They fear thought."
        assert entailment_problem(text, text, self.FakeNLI(), min_fraction=0.9) is None


class TestFinalize:
    def test_writes_splits_and_dataset_info(self, tmp_path):
        from filter_training_data import finalize
        raw = tmp_path / "train.jsonl"
        rows = [_row((i,), idx=i) for i in range(100)]
        raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        lf_dir = tmp_path / "LlamaFactory"
        stats = finalize(raw, lf_dir, dataset_name="russell", val_fraction=0.1, nli=False, seed=3, block_size=5,
                         persona=lambda row: "PERSONA")

        info = json.loads((lf_dir / "dataset_info.json").read_text())
        assert info["russell_sft"]["file_name"] == "train.jsonl"
        assert info["russell_val"]["file_name"] == "val.jsonl"
        train = [json.loads(l) for l in (lf_dir / "train.jsonl").read_text().splitlines()]
        val = [json.loads(l) for l in (lf_dir / "val.jsonl").read_text().splitlines()]
        assert len(train) == stats["train"] and len(val) == stats["val"] and val
        # LlamaFactory rows keep only the columns dataset_info maps.
        assert set(train[0]) == {"system", "input", "output"}
