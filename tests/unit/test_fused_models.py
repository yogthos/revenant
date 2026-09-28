"""Tests for fused model support: fusion script, CLI parsing, config wiring."""

import argparse
import inspect
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestFuseMLXActuallyFuses:
    """Priority 1: fuse_mlx must merge LoRA weights into base weights.

    Regression test for a bug where `remove_lora_layers` was called instead of
    `m.fuse()`. `remove_lora_layers` strips the LoRALinear wrapper but returns
    the original base linear unchanged — the LoRA weights are silently lost.
    """

    def test_fuse_mlx_does_not_invoke_remove_lora_layers(self):
        """fuse_mlx must not call remove_lora_layers (which doesn't actually fuse)."""
        import scripts.fuse_model as fm

        source = inspect.getsource(fm.fuse_mlx)
        # Strip line comments so we match on executable code only
        code_only = "\n".join(
            line.split("#", 1)[0] for line in source.splitlines()
        )
        assert "remove_lora_layers(" not in code_only, (
            "fuse_mlx must not call remove_lora_layers — it strips LoRA layers "
            "without merging their weights into the base."
        )

    def test_fuse_mlx_uses_fuse_method(self):
        """fuse_mlx must call .fuse() on LoRA modules (canonical mlx-lm pattern)."""
        import scripts.fuse_model as fm

        source = inspect.getsource(fm.fuse_mlx)
        assert ".fuse(" in source or "m.fuse" in source, (
            "fuse_mlx must call .fuse() on modules with LoRA weights "
            "to actually merge them into base weights."
        )

    def test_fuse_mlx_updates_modules(self):
        """fuse_mlx must replace LoRA modules with fused linears via update_modules."""
        import scripts.fuse_model as fm

        source = inspect.getsource(fm.fuse_mlx)
        assert "update_modules" in source, (
            "fuse_mlx must call model.update_modules() to install the fused layers."
        )


class TestFuseMLXScaleOverride:
    """fuse_mlx must accept a scale override that replaces the trained scale
    on every LoRA module before fusing."""

    def test_fuse_mlx_signature_accepts_scale(self):
        """fuse_mlx must expose a `scale` keyword argument."""
        import inspect
        import scripts.fuse_model as fm

        sig = inspect.signature(fm.fuse_mlx)
        assert "scale" in sig.parameters, (
            "fuse_mlx must accept a `scale` kwarg for overriding the trained scale"
        )

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("mlx_lm") is None,
        reason="mlx_lm is Apple-only; test patches its symbols",
    )
    def test_fuse_mlx_overrides_module_scale_before_fusing(self):
        """When scale is passed, every LoRA module's .scale is reassigned
        before .fuse() is invoked."""
        from unittest.mock import MagicMock, patch

        # Build fake LoRALinear-ish modules that record their state at .fuse() time
        captured_scales = []

        def make_module():
            m = MagicMock()
            m.scale = 2.0  # original trained scale
            m.fuse = MagicMock(side_effect=lambda dequantize=False: captured_scales.append(m.scale) or MagicMock())
            return m

        lora_modules = [make_module() for _ in range(3)]
        fake_model = MagicMock()
        fake_model.named_modules = MagicMock(
            return_value=[(f"layer{i}", m) for i, m in enumerate(lora_modules)]
        )
        fake_tokenizer = MagicMock()
        fake_config = {}

        with patch("mlx_lm.utils.load", return_value=(fake_model, fake_tokenizer, fake_config)), \
             patch("mlx_lm.utils.save"), \
             patch("src.generation.lora_generator.check_adapter_loaded"), \
             patch("mlx.utils.tree_unflatten"):
            from scripts.fuse_model import fuse_mlx

            tmp_adapter = Path(tempfile.mkdtemp())
            tmp_output = Path(tempfile.mkdtemp())
            try:
                fuse_mlx(
                    model_path="dummy",
                    adapter_path=tmp_adapter,
                    output_path=tmp_output,
                    scale=0.5,
                )
            finally:
                import shutil
                shutil.rmtree(tmp_adapter, ignore_errors=True)
                shutil.rmtree(tmp_output, ignore_errors=True)

        # Every module must have had scale==0.5 at the moment .fuse() was called
        assert captured_scales == [0.5, 0.5, 0.5], (
            f"scale override not applied before fuse; saw {captured_scales}"
        )

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("mlx_lm") is None,
        reason="mlx_lm is Apple-only; test patches its symbols",
    )
    def test_fuse_mlx_leaves_scale_alone_when_not_overridden(self):
        """When scale=None, modules keep their trained scale."""
        from unittest.mock import MagicMock, patch

        captured_scales = []

        def make_module():
            m = MagicMock()
            m.scale = 2.0
            m.fuse = MagicMock(side_effect=lambda dequantize=False: captured_scales.append(m.scale) or MagicMock())
            return m

        lora_modules = [make_module() for _ in range(2)]
        fake_model = MagicMock()
        fake_model.named_modules = MagicMock(
            return_value=[(f"layer{i}", m) for i, m in enumerate(lora_modules)]
        )

        with patch("mlx_lm.utils.load", return_value=(fake_model, MagicMock(), {})), \
             patch("mlx_lm.utils.save"), \
             patch("src.generation.lora_generator.check_adapter_loaded"), \
             patch("mlx.utils.tree_unflatten"):
            from scripts.fuse_model import fuse_mlx

            tmp_adapter = Path(tempfile.mkdtemp())
            tmp_output = Path(tempfile.mkdtemp())
            try:
                fuse_mlx(
                    model_path="dummy",
                    adapter_path=tmp_adapter,
                    output_path=tmp_output,
                    scale=None,
                )
            finally:
                import shutil
                shutil.rmtree(tmp_adapter, ignore_errors=True)
                shutil.rmtree(tmp_output, ignore_errors=True)

        assert captured_scales == [2.0, 2.0]


class TestScaleCLIFlag:
    """--scale CLI flag must parse to a float and reach fuse_mlx."""

    def test_scale_flag_reaches_fuse_mlx(self):
        """`--scale 0.75` should reach fuse_mlx as scale=0.75."""
        from unittest.mock import patch
        import sys

        captured = {}

        def fake_fuse_mlx(*args, **kwargs):
            captured.update(kwargs)

        tmp_checkpoint = Path(tempfile.mkdtemp())
        tmp_output = Path(tempfile.mkdtemp())
        try:
            with patch("scripts.fuse_model.fuse_mlx", side_effect=fake_fuse_mlx), \
                 patch.object(
                     sys,
                     "argv",
                     [
                         "fuse_model.py",
                         "--model", "dummy",
                         "--checkpoint", str(tmp_checkpoint),
                         "--output", str(tmp_output),
                         "--mlx",
                         "--scale", "0.75",
                     ],
                 ):
                from scripts.fuse_model import main
                try:
                    main()
                except SystemExit:
                    pass
        finally:
            import shutil
            shutil.rmtree(tmp_checkpoint, ignore_errors=True)
            shutil.rmtree(tmp_output, ignore_errors=True)

        assert captured.get("scale") == 0.75, (
            f"--scale 0.75 did not reach fuse_mlx; captured={captured}"
        )


class TestModelCLIAppend:
    """Priority 2: --model must be action='append' so it yields a list of paths."""

    def test_single_model_produces_list(self):
        """One --model flag should produce a single-element list, not a string."""
        from restyle import _build_argument_parser

        parser = _build_argument_parser()
        args = parser.parse_args(
            ["input.md", "-o", "out.md", "--model", "models/foo"]
        )
        assert args.model == ["models/foo"]

    def test_multiple_model_flags_collected(self):
        """Multiple --model flags should produce a list of all paths."""
        from restyle import _build_argument_parser

        parser = _build_argument_parser()
        args = parser.parse_args(
            [
                "input.md",
                "-o",
                "out.md",
                "--model",
                "models/foo",
                "--model",
                "models/bar",
            ]
        )
        assert args.model == ["models/foo", "models/bar"]

    def test_no_model_flag_is_none_or_empty(self):
        """No --model flag should leave args.model falsy (None or [])."""
        from restyle import _build_argument_parser

        parser = _build_argument_parser()
        args = parser.parse_args(["input.md", "-o", "out.md", "--adapter", "foo"])
        assert not args.model


class TestGetFusedModelConfig:
    """Priority 3: get_fused_model_config must look up per-model settings by path."""

    def test_get_fused_model_config_returns_defaults_for_none(self):
        """Passing None should return a default FusedModelConfig."""
        from src.config import FusedModelConfig, get_fused_model_config

        cfg = get_fused_model_config(None)
        assert isinstance(cfg, FusedModelConfig)
        # Default values from dataclass
        assert cfg.enabled is True
        assert cfg.temperature == 0.6

    def test_get_fused_model_config_by_exact_path(self):
        """Exact-path match should return that model's config."""
        from src.config import _config_cache, get_fused_model_config, load_config

        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "use_adapter": False,
                "models": {
                    "models/foo": {
                        "author": "Foo Author",
                        "temperature": 0.42,
                        "worldview": "foo.txt",
                    }
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            try:
                _config_cache.clear()
                # Prime the default-path cache entry with our test config so the
                # internal load_config() call (no args) returns the right data.
                config = load_config(f.name)
                _config_cache["config.json"] = config

                cfg = get_fused_model_config("models/foo")
                assert cfg.author == "Foo Author"
                assert cfg.temperature == 0.42
                assert cfg.worldview == "foo.txt"
            finally:
                _config_cache.clear()
                os.unlink(f.name)

    def test_get_fused_model_config_by_directory_name(self):
        """Should match by trailing directory name if exact path not in config."""
        from src.config import _config_cache, get_fused_model_config, load_config

        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "use_adapter": False,
                "models": {
                    "models/my-fused-model": {
                        "author": "Bar",
                        "worldview": "bar.txt",
                    }
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            try:
                _config_cache.clear()
                config = load_config(f.name)
                _config_cache["config.json"] = config

                cfg = get_fused_model_config("/other/prefix/my-fused-model")
                assert cfg.author == "Bar"
                assert cfg.worldview == "bar.txt"
            finally:
                _config_cache.clear()
                os.unlink(f.name)


class TestWorldviewLookupFusedModels:
    """Priority 4: worldview lookup must also check the fused `models` dict."""

    def test_worldview_found_for_fused_model_path(self):
        """Passing a fused-model path should resolve its worldview field."""
        from src.config import _config_cache, load_config
        from src.persona.prompt_builder import _get_worldview_filename

        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "use_adapter": False,
                "models": {
                    "models/my-fused": {
                        "author": "Q",
                        "worldview": "q_worldview.txt",
                    }
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            f.flush()
            try:
                _config_cache.clear()
                config = load_config(f.name)
                _config_cache["config.json"] = config

                result = _get_worldview_filename("models/my-fused")
                assert result == "q_worldview.txt"
            finally:
                _config_cache.clear()
                os.unlink(f.name)


class TestTransferAppliesFusedModelSettings:
    """Priority 3 (integration): TransferConfig should pick up FusedModelConfig settings."""

    def test_fused_model_overrides_applied_to_transfer_config(self):
        """When a fused model has perspective/verify_entailment/etc set, TransferConfig
        should adopt those values (mirroring the adapter override path)."""
        from src.config import FusedModelConfig
        from src.generation.transfer import TransferConfig, _apply_fused_model_overrides

        cfg = TransferConfig()
        cfg.expand_for_texture_explicit = False  # Not set by CLI

        fused_cfg = FusedModelConfig(
            author="X",
            perspective="first_person_singular",
            verify_entailment=False,
            expand_for_texture=True,
            merge_paragraphs=200,
        )

        _apply_fused_model_overrides(cfg, fused_cfg)

        assert cfg.perspective == "first_person_singular"
        assert cfg.verify_semantic_fidelity is False
        assert cfg.expand_for_texture is True
        assert cfg.merge_paragraphs == 200

    def test_fused_model_cli_expand_takes_priority(self):
        """If expand_for_texture_explicit is True, fused cfg should not override."""
        from src.config import FusedModelConfig
        from src.generation.transfer import TransferConfig, _apply_fused_model_overrides

        cfg = TransferConfig()
        cfg.expand_for_texture = False
        cfg.expand_for_texture_explicit = True  # Set by CLI

        fused_cfg = FusedModelConfig(expand_for_texture=True)
        _apply_fused_model_overrides(cfg, fused_cfg)

        # CLI wins
        assert cfg.expand_for_texture is False

    def test_fused_model_cli_no_verify_takes_priority(self):
        """--no-verify must survive a model config that sets verify_entailment=true."""
        from src.config import FusedModelConfig
        from src.generation.transfer import TransferConfig, _apply_fused_model_overrides

        cfg = TransferConfig()
        cfg.verify_semantic_fidelity = False
        cfg.verify_semantic_fidelity_explicit = True  # Set by --no-verify

        fused_cfg = FusedModelConfig(verify_entailment=True)
        _apply_fused_model_overrides(cfg, fused_cfg)

        assert cfg.verify_semantic_fidelity is False

    def test_fused_model_verify_applies_when_not_explicit(self):
        """Without the CLI flag, the model config still drives verification."""
        from src.config import FusedModelConfig
        from src.generation.transfer import TransferConfig, _apply_fused_model_overrides

        cfg = TransferConfig()
        cfg.verify_semantic_fidelity = True
        cfg.verify_semantic_fidelity_explicit = False

        fused_cfg = FusedModelConfig(verify_entailment=False)
        _apply_fused_model_overrides(cfg, fused_cfg)

        assert cfg.verify_semantic_fidelity is False


class TestResolveTransferTargets:
    """Priority 5: _resolve_transfer_targets unifies CLI + config resolution.

    Shared between file-transfer and REPL modes so both behave identically
    when falling back to config.json.
    """

    def _make_args(self, **overrides):
        """Build a minimal args-like object with the fields the resolver reads."""
        from types import SimpleNamespace

        defaults = dict(
            model=None,
            adapters=None,
            lora_scale=None,
            checkpoint=None,
            config="config.json",
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_cli_model_wins_over_config(self):
        """--model on CLI takes priority over any config.json content."""
        from restyle import _resolve_transfer_targets

        args = self._make_args(model=["models/cli-model"])
        adapters, fused, fused_cfg = _resolve_transfer_targets(args)
        assert fused == ["models/cli-model"]
        assert adapters == []

    def test_config_fused_models_when_use_adapter_false(self):
        """With use_adapter=false, resolver returns enabled fused models."""
        from src.config import _config_cache, load_config
        from restyle import _resolve_transfer_targets

        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "use_adapter": False,
                "models": {
                    "models/enabled": {
                        "enabled": True,
                        "author": "X",
                        "worldview": "x.txt",
                    },
                    "models/disabled": {
                        "enabled": False,
                        "author": "Y",
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            try:
                _config_cache.clear()
                load_config(f.name)  # populate cache under f.name

                args = self._make_args(config=f.name)
                adapters, fused, fused_cfg = _resolve_transfer_targets(args)
                assert fused == ["models/enabled"]
                assert adapters == []
                assert fused_cfg is not None
                assert fused_cfg.author == "X"
            finally:
                _config_cache.clear()
                os.unlink(f.name)

    def test_config_lora_adapters_when_use_adapter_true(self):
        """With use_adapter=true, resolver returns enabled adapters."""
        from src.config import _config_cache, load_config
        from restyle import _resolve_transfer_targets

        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "use_adapter": True,
                "lora_adapters": {
                    "lora_adapters/foo": {
                        "enabled": True,
                        "scale": 2.0,
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            try:
                _config_cache.clear()
                load_config(f.name)

                args = self._make_args(config=f.name)
                adapters, fused, fused_cfg = _resolve_transfer_targets(args)
                assert fused == []
                assert len(adapters) == 1
                assert adapters[0].path == "lora_adapters/foo"
                assert adapters[0].scale == 2.0
            finally:
                _config_cache.clear()
                os.unlink(f.name)


class TestReplWithFusedModel:
    """Priority 5: REPL mode must accept --model and use fused models from config."""

    def test_run_repl_passes_fused_models_to_transfer(self):
        """run_repl with fused_models set should construct StyleTransfer with fused_models."""
        from unittest.mock import MagicMock, patch

        with patch("src.repl.repl.StyleTransfer") as mock_transfer, patch(
            "src.repl.repl.StyleREPL"
        ) as mock_repl:
            # Make the REPL a no-op
            mock_repl.return_value.run = MagicMock()

            from src.repl.repl import run_repl

            run_repl(
                adapter_path=None,
                author="Test",
                fused_models=["models/my-fused"],
                verify=False,
            )

            # StyleTransfer should have been constructed with fused_models
            assert mock_transfer.call_count == 1
            kwargs = mock_transfer.call_args.kwargs
            assert kwargs.get("fused_models") == ["models/my-fused"]

    def test_run_repl_signature_accepts_fused_models(self):
        """run_repl must have a fused_models keyword parameter."""
        import inspect
        from src.repl.repl import run_repl

        sig = inspect.signature(run_repl)
        assert "fused_models" in sig.parameters, (
            "run_repl must accept fused_models so REPL can use fused models"
        )


class TestSampleConfigIsClean:
    """config.json.sample must parse without unknown-field warnings."""

    def test_sample_config_loads_without_unknown_field_warnings(self):
        """The checked-in sample must only contain keys the parser knows about."""
        from src.config import _config_cache, load_config

        sample_path = Path(__file__).parent.parent.parent / "config.json.sample"
        assert sample_path.exists(), "config.json.sample not found"

        _config_cache.clear()
        try:
            with patch("src.config.logger") as mock_logger:
                load_config(str(sample_path))
                warnings = [str(c) for c in mock_logger.warning.call_args_list]
                unknown_warnings = [
                    w for w in warnings if "Unknown" in w and "fields" in w
                ]
                assert not unknown_warnings, (
                    f"config.json.sample has unknown fields: {unknown_warnings}"
                )
        finally:
            _config_cache.clear()

    def test_sample_config_has_models_and_lora_adapters(self):
        """The sample must demonstrate both adapter and fused-model configuration."""
        from src.config import _config_cache, load_config

        sample_path = Path(__file__).parent.parent.parent / "config.json.sample"
        _config_cache.clear()
        try:
            config = load_config(str(sample_path))
            assert config.generation.lora_adapters, "sample lacks lora_adapters examples"
            assert config.generation.models, "sample lacks fused-model examples"
        finally:
            _config_cache.clear()


class TestGenerationConfigFromFusedModel:
    """Priority 3: GenerationConfig.from_fused_model builds from FusedModelConfig."""

    def test_from_fused_model_applies_all_gen_settings(self):
        """from_fused_model should copy temperature, top_p, min_p, etc. from FusedModelConfig."""
        from src.config import FusedModelConfig
        from src.generation.base_generator import GenerationConfig

        fused = FusedModelConfig(
            temperature=0.42,
            top_p=0.88,
            min_p=0.07,
            repetition_penalty=1.22,
            max_tokens=1234,
        )
        gen = GenerationConfig.from_fused_model(fused)
        assert gen.temperature == 0.42
        assert gen.top_p == 0.88
        assert gen.min_p == 0.07
        assert gen.repetition_penalty == 1.22
        assert gen.max_tokens == 1234


class TestFusedModelKeepsTrainingTemplate:
    """A fused model has no adapter, so the template recorded in the adapter's
    metadata.json has to travel with it or prompts fall back to the default."""

    ADAPTER_META = {"author": "Bertrand Russell", "base_model": "models/Hemmingway-1-6bit-MLX",
                    "lora_rank": 64, "lora_alpha": 64, "template": "qwen3_8",
                    "enable_thinking": False, "persona_turn": "system"}

    def test_fuse_copies_the_adapter_metadata(self, tmp_path):
        import scripts.fuse_model as fm
        adapter, out = tmp_path / "adapter", tmp_path / "fused"
        adapter.mkdir(); out.mkdir()
        (adapter / "metadata.json").write_text(json.dumps(self.ADAPTER_META))
        (adapter / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 64, "scale": 1.0}}))
        fm.write_fuse_metadata(out, adapter, "models/Hemmingway-1", scale=None, qbits=8, group_size=64)
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["template"] == "qwen3_8"
        assert meta["enable_thinking"] is False and meta["persona_turn"] == "system"
        assert meta["base_model"] == "models/Hemmingway-1"
        assert json.loads((out / "fuse_metadata.json").read_text())["quantization_bits"] == 8

    def test_generator_reads_it_in_fused_mode(self, tmp_path):
        pytest.importorskip("mlx_lm")
        from src.generation.lora_generator import LoRAStyleGenerator
        from src.generation.base_generator import GenerationConfig
        (tmp_path / "metadata.json").write_text(json.dumps(self.ADAPTER_META))
        gen = LoRAStyleGenerator(base_model=str(tmp_path), config=GenerationConfig())
        assert gen.metadata.template == "qwen3_8"
        assert gen.metadata.persona_turn == "system"
        assert gen.base_model_name == str(tmp_path)
        assert gen.chat_prompt("PERSONA", "text").startswith("<|im_start|>system\nPERSONA")

    def test_pytorch_generator_reads_it_in_fused_mode(self, tmp_path):
        pytest.importorskip("torch")
        from src.generation.pytorch_generator import PyTorchStyleGenerator
        from src.generation.base_generator import GenerationConfig
        (tmp_path / "metadata.json").write_text(json.dumps(self.ADAPTER_META))
        gen = PyTorchStyleGenerator(base_model=str(tmp_path), config=GenerationConfig())
        assert gen.metadata.template == "qwen3_8" and gen.metadata.persona_turn == "system"

    def test_mlx_conversion_keeps_it(self):
        import scripts.fuse_model as fm
        assert '"metadata.json"' in inspect.getsource(fm.convert_to_mlx)
