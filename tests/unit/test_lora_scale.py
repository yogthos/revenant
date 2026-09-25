"""`scale` in config.json is a multiplier on the strength the adapter was trained at."""

from types import SimpleNamespace

import numpy as np
import pytest


def _lora_module(scale):
    return SimpleNamespace(lora_a=np.ones((4, 2)), lora_b=np.ones((2, 3)), scale=scale)


class TestMLXScale:
    def _generator(self, model):
        from src.generation.lora_generator import LoRAStyleGenerator
        gen = LoRAStyleGenerator.__new__(LoRAStyleGenerator)
        gen._model = model
        return gen

    def test_scale_multiplies_trained_scale(self):
        layer = _lora_module(2.0)
        gen = self._generator(SimpleNamespace(named_modules=lambda: [("", None), ("layers.0.q_proj", layer)]))
        gen._apply_lora_scale(0.5)
        assert layer.scale == pytest.approx(1.0)

    def test_reaches_lora_layers_inside_layer_lists(self):
        # model.layers is a list; the old recursive walk never got past it.
        pytest.importorskip("mlx")
        mlx_nn = pytest.importorskip("mlx.nn")
        from mlx_lm.tuner.lora import LoRALinear

        class Model(mlx_nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [LoRALinear.from_base(mlx_nn.Linear(4, 4), r=2, scale=2.0) for _ in range(2)]

        gen = self._generator(Model())
        gen._apply_lora_scale(1.5)
        assert [layer.scale for layer in gen._model.layers] == [pytest.approx(3.0)] * 2


class TestStackAdapters:
    def _adapter(self, rng, rank, trained_scale, keys=("layers.0.q_proj",)):
        weights = {}
        for key in keys:
            weights[f"{key}.lora_a"] = rng.standard_normal((4, rank))
            weights[f"{key}.lora_b"] = rng.standard_normal((rank, 3))
        return weights, trained_scale

    def test_stacked_adapter_equals_weighted_sum(self):
        from src.generation.lora_generator import stack_lora_adapters
        rng = np.random.default_rng(0)
        (w1, s1), (w2, s2) = self._adapter(rng, 2, 2.0), self._adapter(rng, 3, 1.0)
        stacked, rank = stack_lora_adapters([(w1, s1, 1.0), (w2, s2, 0.5)])
        assert rank == 5
        x = rng.standard_normal((1, 4))
        key = "layers.0.q_proj"
        got = x @ stacked[f"{key}.lora_a"] @ stacked[f"{key}.lora_b"]
        want = (2.0 * 1.0 * x @ w1[f"{key}.lora_a"] @ w1[f"{key}.lora_b"]
                + 1.0 * 0.5 * x @ w2[f"{key}.lora_a"] @ w2[f"{key}.lora_b"])
        np.testing.assert_allclose(got, want, rtol=1e-6)

    def test_module_missing_from_one_adapter_gets_zero_block(self):
        from src.generation.lora_generator import stack_lora_adapters
        rng = np.random.default_rng(1)
        (w1, s1) = self._adapter(rng, 2, 1.0, keys=("layers.0.q_proj", "layers.0.v_proj"))
        (w2, s2) = self._adapter(rng, 2, 1.0, keys=("layers.0.q_proj",))
        stacked, _ = stack_lora_adapters([(w1, s1, 1.0), (w2, s2, 1.0)])
        x = rng.standard_normal((1, 4))
        key = "layers.0.v_proj"
        got = x @ stacked[f"{key}.lora_a"] @ stacked[f"{key}.lora_b"]
        np.testing.assert_allclose(got, x @ w1[f"{key}.lora_a"] @ w1[f"{key}.lora_b"], rtol=1e-6)


class TestPyTorchScale:
    def test_scale_multiplies_trained_scale(self):
        pytest.importorskip("torch")
        from src.generation.pytorch_generator import PyTorchStyleGenerator
        gen = PyTorchStyleGenerator.__new__(PyTorchStyleGenerator)
        module = SimpleNamespace(scaling={"default": 2.0})
        gen._model = SimpleNamespace(named_modules=lambda: [("m", module)])
        gen._apply_adapter_scale(0.5)
        assert module.scaling["default"] == pytest.approx(1.0)
