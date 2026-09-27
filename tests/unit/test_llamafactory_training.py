"""LlamaFactory configs, the row filter and the PEFT -> MLX converter agree with inference."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TRACKED_YAMLS = [
    ROOT / p for p in subprocess.run(
        ["git", "ls-files", "data/training/*/LlamaFactory/*.yaml"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.split()
]
HEMMINGWAY = ROOT / "data/training/russell/LlamaFactory/hemmingway1_27b_lora.yaml"


def _load(path):
    return yaml.safe_load(path.read_text())


class TestTrainingConfigs:
    def test_found_configs(self):
        assert HEMMINGWAY in TRACKED_YAMLS or HEMMINGWAY.exists()
        assert len(TRACKED_YAMLS) >= 5

    @pytest.mark.parametrize("path", TRACKED_YAMLS + [HEMMINGWAY], ids=lambda p: str(p.relative_to(ROOT)))
    def test_config(self, path):
        from filter_training_data import DEFAULT_MAX_TOKENS
        from src.generation.base_generator import render_chat_prompt

        cfg = _load(path)
        # Packed rows attend to each other, and Qwen3.5's linear-attention
        # layers carry their state across them.
        assert not cfg.get("packing"), "packing lets training rows leak into each other"
        # Inference renders this template itself, so it has to be one it knows.
        render_chat_prompt("i", "c", cfg["template"], enable_thinking=cfg.get("enable_thinking", True))
        if path.parent.parent.name == "russell":
            # finalize() drops rows over DEFAULT_MAX_TOKENS, so none get cut.
            assert cfg["cutoff_len"] >= DEFAULT_MAX_TOKENS
        if cfg.get("eval_steps"):
            assert cfg["save_steps"] % cfg["eval_steps"] == 0
        info = json.loads((path.parent / "dataset_info.json").read_text())
        for name in str(cfg["dataset"]).split(","):
            assert name.strip() in info
        if cfg.get("eval_dataset"):
            assert cfg["eval_dataset"] in info

    def test_hemmingway_config(self):
        cfg = _load(HEMMINGWAY)
        assert cfg["model_name_or_path"] == "Altworld/Hemmingway-1"
        # Chat model with a thinking template: train in its non-thinking format.
        assert cfg["template"] == "qwen3_8"
        assert cfg["enable_thinking"] is False
        assert cfg["dataset"] == "russell_sft" and cfg["eval_dataset"] == "russell_val"
        assert cfg["load_best_model_at_end"] is True
        # Persona in the system turn, text in the user turn.
        info = json.loads((HEMMINGWAY.parent / "dataset_info.json").read_text())
        assert info["russell_sft"]["columns"].get("system") == "system"

    def test_hemmingway_trains_in_bf16(self):
        # Unsloth: QLoRA on Qwen3.5 models (dense or MoE) loses more to
        # quantization than usual. bf16 LoRA fits one 80GB card.
        cfg = _load(HEMMINGWAY)
        assert "quantization_bit" not in cfg
        assert cfg["bf16"] is True

    def test_hemmingway_adapter_scale_is_one(self):
        # The converter bakes this in and config.json's scale 1.0 means "as trained".
        cfg = _load(HEMMINGWAY)
        rank, alpha = cfg["lora_rank"], cfg["lora_alpha"]
        scale = alpha / rank ** 0.5 if cfg.get("use_rslora") else alpha / rank
        assert scale == 1.0

    def test_hemmingway_run_resumes_after_a_crash(self):
        # With overwrite_output_dir LlamaFactory ignores existing checkpoints,
        # so relaunching a multi-hour headless run would start over.
        cfg = _load(HEMMINGWAY)
        assert cfg.get("overwrite_output_dir") is False
        assert not cfg.get("save_only_model")
        assert cfg["save_strategy" if "save_strategy" in cfg else "eval_strategy"] == "steps"


class TestArchiveAdapters:
    SCRIPT = ROOT / "scripts/runpod/archive_adapters.sh"

    def _checkpoint(self, root, step, complete=True):
        ckpt = root / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt"):
            (ckpt / name).write_text(name)
        if complete:
            (ckpt / "trainer_state.json").write_text("{}")
        return ckpt

    def test_copies_adapters_of_finished_checkpoints(self, tmp_path):
        saves, archive = tmp_path / "saves", tmp_path / "adapters"
        self._checkpoint(saves, 100)
        self._checkpoint(saves, 200, complete=False)
        subprocess.run(["bash", str(self.SCRIPT), str(saves), str(archive), "--once"], check=True)
        assert (archive / "checkpoint-100/adapter_model.safetensors").exists()
        assert (archive / "checkpoint-100/trainer_state.json").exists()
        assert not (archive / "checkpoint-100/optimizer.pt").exists()
        assert not (archive / "checkpoint-200").exists()


class TestRowLength:
    def test_rows_over_the_token_budget_are_dropped(self):
        from filter_training_data import row_problem
        inp = "The man walked to the town and bought some bread for his family. " * 4
        out = "He went into town, as he did most days, and came back with bread. " * 4
        row = {"instruction": "Frame.", "input": inp, "output": out}
        assert row_problem(row, max_tokens=2048) is None
        long_row = {"instruction": "Frame. " * 2000, "input": inp, "output": out}
        assert "too long" in row_problem(long_row, max_tokens=2048)

    def test_estimate_is_conservative(self):
        from filter_training_data import estimate_tokens
        # ~4.6 chars per Qwen token on this corpus; the estimate may not undercount.
        text = "The present chapter deals with the analysis of matter, and of events. " * 20
        assert estimate_tokens(text) >= len(text) / 4.6


class TestConverter:
    @pytest.fixture
    def peft_dir(self, tmp_path):
        torch = pytest.importorskip("torch")
        from safetensors.torch import save_file
        peft = tmp_path / "peft"
        peft.mkdir()
        save_file({
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, 8),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(8, 4),
        }, str(peft / "adapter_model.safetensors"))
        (peft / "adapter_config.json").write_text(json.dumps(
            {"r": 4, "lora_alpha": 8, "use_rslora": False, "base_model_name_or_path": "Altworld/Hemmingway-1"}))
        return peft

    def _mlx_model(self, tmp_path, keys):
        pytest.importorskip("torch")
        import numpy as np
        from safetensors.numpy import save_file
        model = tmp_path / "mlx"
        model.mkdir()
        save_file({k: np.zeros((1,), dtype=np.float32) for k in keys}, str(model / "model.safetensors"))
        return model

    def test_records_training_template_and_scale(self, tmp_path, peft_dir):
        from convert_peft_to_mlx import convert_peft_to_mlx
        mlx = self._mlx_model(tmp_path, ["language_model.model.layers.0.self_attn.q_proj.weight"])
        out = tmp_path / "out"
        convert_peft_to_mlx(peft_dir, out, mlx_model_path=str(mlx), train_config=HEMMINGWAY)
        meta = json.loads((out / "metadata.json").read_text())
        assert (meta["template"], meta["enable_thinking"], meta["persona_turn"]) == ("qwen3_8", False, "system")
        adapter = json.loads((out / "adapter_config.json").read_text())
        assert adapter["lora_parameters"]["scale"] == pytest.approx(2.0)
        from safetensors.numpy import load_file
        assert "language_model.model.layers.0.self_attn.q_proj.lora_a" in load_file(str(out / "adapters.safetensors"))

    def test_weights_the_model_lacks_are_an_error(self, tmp_path, peft_dir):
        # mlx_lm loads adapters with strict=False, so a key that matches no
        # module would be dropped without a word.
        from convert_peft_to_mlx import convert_peft_to_mlx
        mlx = self._mlx_model(tmp_path, ["language_model.model.layers.0.self_attn.k_proj.weight"])
        with pytest.raises(ValueError, match="q_proj"):
            convert_peft_to_mlx(peft_dir, tmp_path / "out", mlx_model_path=str(mlx), train_config=HEMMINGWAY)


class TestTextOnlyTemplate:
    def test_qwen3_8_encodes_without_an_image_processor(self):
        pytest.importorskip("llamafactory")
        sys.path.insert(0, str(ROOT / "scripts/runpod"))
        from lf_train import use_text_plugin
        from llamafactory.data.template import TEMPLATES
        use_text_plugin()
        TEMPLATES["qwen3_8"].mm_plugin._validate_input(None, [], [], [])


class TestLigerAlias:
    """LlamaFactory imports apply_liger_kernel_to_qwen3_5_text, which no
    liger-kernel release has; the qwen3_5 patch covers Qwen3_5ForCausalLM."""

    @pytest.fixture
    def fake_liger(self, monkeypatch):
        import types
        from unittest.mock import MagicMock
        pkg, mod = types.ModuleType("liger_kernel"), types.ModuleType("liger_kernel.transformers")
        mod.apply_liger_kernel_to_qwen3_5 = MagicMock()
        pkg.transformers = mod
        monkeypatch.setitem(sys.modules, "liger_kernel", pkg)
        monkeypatch.setitem(sys.modules, "liger_kernel.transformers", mod)
        sys.path.insert(0, str(ROOT / "scripts/runpod"))
        return mod

    def test_text_model_gets_only_the_fused_loss(self, fake_liger):
        import inspect
        from lf_train import alias_liger_qwen3_5_text
        alias_liger_qwen3_5_text()
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5_text as apply
        # LlamaFactory checks for this parameter before calling with no kwargs.
        assert "fused_linear_cross_entropy" in inspect.signature(apply).parameters
        apply()
        kwargs = fake_liger.apply_liger_kernel_to_qwen3_5.call_args.kwargs
        assert kwargs["fused_linear_cross_entropy"] is True
        assert kwargs["rms_norm"] is False and kwargs["swiglu"] is False

    def test_keeps_a_real_implementation(self, fake_liger):
        from lf_train import alias_liger_qwen3_5_text
        fake_liger.apply_liger_kernel_to_qwen3_5_text = real = object()
        alias_liger_qwen3_5_text()
        assert fake_liger.apply_liger_kernel_to_qwen3_5_text is real
