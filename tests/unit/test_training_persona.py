"""Training rows get the same persona instruction inference builds, in the system turn."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OWN = ("The man who has no tincture of philosophy goes through life imprisoned in the prejudices "
       "derived from common sense, from the habitual beliefs of his age or his nation.")
OTHER = ("Philosophy is to be studied not for the sake of any definite answers to its questions, "
         "but rather for the sake of the questions themselves.")


class TestGrafterExclusion:
    def _grafter(self, results):
        from src.rag.structural_grafter import StructuralGrafter
        grafter = StructuralGrafter.__new__(StructuralGrafter)
        grafter.author = "A"
        grafter.indexer = MagicMock()
        grafter.indexer.retrieve_similar.return_value = results
        grafter.llm_provider = None
        grafter._skeleton_cache = {}
        return grafter

    def test_skips_the_rows_own_paragraph(self):
        grafter = self._grafter([{"text": OWN, "skeleton": "[Own] -> [Moves]"},
                                 {"text": OTHER, "skeleton": "[Other] -> [Moves]"}])
        guidance = grafter.get_grafting_guidance("neutral input", exclude=OWN[:120])
        assert guidance.sample_text == OTHER
        assert guidance.skeleton.moves == ["Other", "Moves"]

    def test_without_exclude_takes_the_best_match(self):
        grafter = self._grafter([{"text": OWN, "skeleton": "[Own] -> [Moves]"}])
        assert grafter.get_grafting_guidance("neutral input").sample_text == OWN
        assert grafter.indexer.retrieve_similar.call_args.kwargs["n"] == 1

    def test_on_the_fly_skeletons_are_cached(self, monkeypatch):
        from src.rag import structural_grafter as sg
        from src.rag.skeleton_extractor import ArgumentSkeleton
        grafter = self._grafter([{"text": OTHER, "skeleton": ""}])
        grafter.llm_provider = object()
        extract = MagicMock(return_value=ArgumentSkeleton(moves=["Claim"], raw="[Claim]"))
        monkeypatch.setattr(sg, "extract_skeleton", extract)
        grafter.get_grafting_guidance("a")
        grafter.get_grafting_guidance("b")
        assert extract.call_count == 1


class TestPersonaBuilder:
    def test_builds_the_inference_instruction_from_the_row(self, monkeypatch):
        import filter_training_data as ftd
        build = MagicMock(return_value="PERSONA")
        monkeypatch.setattr(ftd, "build_persona_instruction", build)
        rag, grafter = MagicMock(), MagicMock()
        rag.get_guidance.return_value.format_for_prompt.return_value = "Rhythm: LONG"
        grafter.get_grafting_guidance.return_value = "GRAFT"
        persona = ftd.PersonaBuilder(worldview="russell_worldview.txt", rag=rag, grafter=grafter)

        row = {"input": "neutral words here", "output": "one two three four five"}
        assert persona(row) == "PERSONA"
        # Guidance is looked up from what the model sees, like inference does
        # with the user's paragraph, and never grafts the row's own paragraph.
        rag.get_guidance.assert_called_once_with("neutral words here")
        assert grafter.get_grafting_guidance.call_args.args[0] == "neutral words here"
        assert grafter.get_grafting_guidance.call_args.kwargs["exclude"] == row["output"]
        kwargs = build.call_args.kwargs
        assert kwargs["structural_guidance"] == "Rhythm: LONG"
        assert kwargs["grafting_guidance"] == "GRAFT"
        assert kwargs["target_words"] == 5
        assert kwargs["worldview"] == "russell_worldview.txt"
        assert kwargs["satisfied_by"] == row["output"]


class TestFinalizeLayout:
    def _rows(self, n=60):
        out = "He went into town, as he did most days, and came back with bread for all. "
        inp = "The man walked to the town and bought some bread for his whole family. "
        return [{"instruction": "old", "input": inp * 2, "output": out * 2, "source_idx": i,
                 "source_paragraphs": [i]} for i in range(n)]

    def test_persona_goes_in_the_system_column(self, tmp_path):
        from filter_training_data import finalize
        raw = tmp_path / "train.jsonl"
        raw.write_text("\n".join(json.dumps(r) for r in self._rows()) + "\n")
        lf = tmp_path / "LlamaFactory"
        finalize(raw, lf, "russell", nli=False, block_size=5, val_fraction=0.1,
                 persona=lambda row: f"PERSONA {row['source_idx']}")
        train = [json.loads(line) for line in (lf / "train.jsonl").read_text().splitlines()]
        assert set(train[0]) == {"system", "input", "output"}
        assert train[0]["system"].startswith("PERSONA ")
        info = json.loads((lf / "dataset_info.json").read_text())
        assert info["russell_sft"]["columns"] == {"prompt": "input", "response": "output", "system": "system"}
        assert info["russell_val"]["columns"] == info["russell_sft"]["columns"]

    def test_persona_is_required(self, tmp_path):
        from filter_training_data import finalize
        raw = tmp_path / "train.jsonl"
        raw.write_text(json.dumps(self._rows(1)[0]) + "\n")
        with pytest.raises(TypeError):
            finalize(raw, tmp_path / "LlamaFactory", "russell", nli=False)

    def test_token_budget_counts_the_system_prompt(self):
        from filter_training_data import row_problem
        row = self._rows(1)[0]
        assert row_problem(row, max_tokens=2048) is None
        assert "too long" in row_problem(dict(row, system="Frame. " * 2000), max_tokens=2048)


class TestGenerationRows:
    def test_llama_factory_rows_leave_the_persona_to_finalize(self):
        import generate_flat_training as gft
        row = gft.format_training_example(neutral_text="Neutral text here.", styled_text="Styled text here.",
                                          author="Bertrand Russell", word_count=3)
        assert set(row) == {"input", "output"}


class TestConverterPersonaTurn:
    def test_reads_the_persona_turn_from_dataset_info(self):
        from convert_peft_to_mlx import read_train_config
        cfg = read_train_config(ROOT / "data/training/russell/LlamaFactory/hemmingway1_27b_lora.yaml")
        assert cfg == {"template": "qwen3_8", "enable_thinking": False, "persona_turn": "system"}

    def test_alpaca_without_system_column_is_user_turn(self, tmp_path):
        from convert_peft_to_mlx import read_train_config
        (tmp_path / "t.yaml").write_text("dataset: d_sft\ntemplate: qwen\n")
        (tmp_path / "dataset_info.json").write_text(json.dumps(
            {"d_sft": {"file_name": "t.jsonl", "columns": {"prompt": "instruction", "query": "input",
                                                           "response": "output"}}}))
        assert read_train_config(tmp_path / "t.yaml")["persona_turn"] == "user"


class TestNLICache:
    """Adding rows shouldn't re-run the entailment check on the old ones."""

    def _rows(self, n=6):
        out = "He went into town, as he did most days, and came back with bread for all. "
        inp = "The man walked to the town and bought some bread for his whole family. "
        return [{"input": inp * 2 + str(i), "output": out * 2 + str(i), "source_idx": i,
                 "source_paragraphs": [i]} for i in range(n)]

    def test_cached_rows_are_not_checked_again(self, tmp_path, monkeypatch):
        import filter_training_data as ftd
        calls = []

        def fake(inp, out, nli, min_fraction=0.75):
            calls.append(inp)
            return "target adds content (x)" if inp.endswith("3") else None

        monkeypatch.setattr(ftd, "entailment_problem", fake)
        raw = tmp_path / "train.jsonl"
        raw.write_text("".join(json.dumps(r) + "\n" for r in self._rows()))
        cache = tmp_path / "nli_cache.json"
        kw = dict(persona=lambda r: "P", nli_model=object(), nli_cache=cache, val_fraction=0.2, block_size=1)
        first = ftd.finalize(raw, tmp_path / "lf", "d", **kw)
        assert len(calls) == 6 and cache.exists()

        rows = self._rows(8)
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
        second = ftd.finalize(raw, tmp_path / "lf", "d", **kw)
        assert len(calls) == 8  # only the two new rows
        assert second["rejected"] == {"target adds content": 1}
        assert second["train"] + second["val"] == first["train"] + first["val"] + 2

    def test_cache_depends_on_the_threshold(self, tmp_path, monkeypatch):
        import filter_training_data as ftd
        calls = []
        monkeypatch.setattr(ftd, "entailment_problem", lambda i, o, n, min_fraction=0.75: calls.append(1))
        raw = tmp_path / "train.jsonl"
        raw.write_text("".join(json.dumps(r) + "\n" for r in self._rows(2)))
        kw = dict(persona=lambda r: "P", nli_model=object(), nli_cache=tmp_path / "c.json")
        ftd.finalize(raw, tmp_path / "lf", "d", **kw)
        ftd.finalize(raw, tmp_path / "lf", "d", nli_min_fraction=0.6, **kw)
        assert len(calls) == 4


class TestLLMStyleMix:
    """--llm-style-share keeps every llm_style row and samples the rest down."""

    def _rows(self, n_llm, n_other):
        out = "He went into town, as he did most days, and came back with bread for all. "
        inp = "The man walked to the town and bought some bread for his whole family. "
        rows = []
        for i in range(n_llm + n_other):
            vtype = "llm_style" if i < n_llm else ("standard" if i % 2 else "snowflake")
            rows.append({"input": inp * 2 + str(i), "output": out * 2 + str(i), "source_idx": i,
                         "source_paragraphs": [i], "variation_type": vtype})
        return rows

    def test_keeps_all_llm_style_rows_and_samples_the_rest(self):
        from filter_training_data import mix_llm_style
        rows = self._rows(70, 200)
        mixed = mix_llm_style(rows, share=0.7, seed=1)
        llm = [r for r in mixed if r["variation_type"] == "llm_style"]
        assert len(llm) == 70
        assert len(mixed) - len(llm) == 30

    def test_keeps_the_original_row_order(self):
        from filter_training_data import mix_llm_style
        rows = self._rows(10, 50)
        mixed = mix_llm_style(rows, share=0.5, seed=1)
        idx = [r["source_idx"] for r in mixed]
        assert idx == sorted(idx)

    def test_same_seed_same_sample(self):
        from filter_training_data import mix_llm_style
        rows = self._rows(10, 50)
        assert mix_llm_style(rows, 0.5, seed=3) == mix_llm_style(rows, 0.5, seed=3)

    def test_too_few_other_rows_keeps_them_all(self):
        from filter_training_data import mix_llm_style
        rows = self._rows(10, 2)
        assert mix_llm_style(rows, share=0.5, seed=1) == rows

    def test_share_must_be_a_fraction(self):
        from filter_training_data import mix_llm_style
        with pytest.raises(ValueError):
            mix_llm_style(self._rows(1, 1), share=1.0)

    def test_finalize_mixes_before_the_split(self, tmp_path):
        from filter_training_data import finalize
        raw = tmp_path / "train.jsonl"
        raw.write_text("".join(json.dumps(r) + "\n" for r in self._rows(20, 80)))
        stats = finalize(raw, tmp_path / "lf", "d", persona=lambda r: "P", nli=False,
                         block_size=1, val_fraction=0.1, llm_style_share=0.5)
        assert stats["train"] + stats["val"] + stats["straddling"] == 40
        assert stats["mixed_out"] == 60


class TestShuffleInputs:
    """--shuffle-share reorders the input sentences of rows that keep the author's order."""

    SENTS = ["The man walked slowly to the old town market.", "He bought fresh bread for his whole family.",
             "Then he carried it home along the river path.", "His children were waiting at the garden gate.",
             "They ate the bread together before the evening meal."]
    OUT = ("Going down to the market in the old town, as was his habit, he came away with loaves enough "
           "for everyone; and when he had brought them home by the river, his children, who had waited "
           "for him at the gate, shared them with him before supper.")

    def _rows(self, n=40):
        rows = []
        for i in range(n):
            vtype = "llm_style" if i % 4 == 0 else "standard"
            rows.append({"input": " ".join(self.SENTS) + f" Row {i} ends here now.",
                         "output": self.OUT + f" So ended day {i}.",
                         "source_idx": i, "source_paragraphs": [i], "variation_type": vtype})
        return rows

    def test_shuffles_that_share_of_non_llm_rows(self):
        from filter_training_data import shuffle_inputs
        rows = self._rows()
        out, n = shuffle_inputs([dict(r) for r in rows], share=0.5, seed=1)
        changed = [o for o, r in zip(out, rows) if o["input"] != r["input"]]
        assert n == len(changed)
        assert 10 <= n <= 20  # about half of the 30 standard rows
        assert all(o["variation_type"] == "standard" for o in changed)
        assert all(sorted(o["input"].split()) == sorted(r["input"].split()) for o, r in zip(out, rows))

    def test_outputs_are_untouched(self):
        from filter_training_data import shuffle_inputs
        rows = self._rows()
        out, _ = shuffle_inputs([dict(r) for r in rows], share=1.0, seed=1)
        assert [o["output"] for o in out] == [r["output"] for r in rows]

    def test_finalize_checks_entailment_before_shuffling(self, tmp_path, monkeypatch):
        import filter_training_data as ftd
        seen = []
        monkeypatch.setattr(ftd, "entailment_problem",
                            lambda i, o, n, min_fraction=0.75: seen.append(i) or None)
        raw = tmp_path / "train.jsonl"
        rows = self._rows(8)
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
        stats = ftd.finalize(raw, tmp_path / "lf", "d", persona=lambda r: "P", nli_model=object(),
                             block_size=1, val_fraction=0.2, shuffle_share=1.0)
        assert sorted(seen) == sorted(r["input"] for r in rows)
        assert stats["shuffled"] == 6
