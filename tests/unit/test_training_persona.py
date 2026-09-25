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
