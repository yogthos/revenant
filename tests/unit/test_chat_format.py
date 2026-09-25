"""Inference prompts must match what LlamaFactory rendered for each training template."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.generation.base_generator import QWEN3_8_XHIGH_REASONING, GenerationConfig, render_chat_prompt

USER = "<|im_start|>user\nFrame.\nNeutral text.<|im_end|>\n<|im_start|>assistant\n"
EMPTY_THOUGHT = "<think>\n\n</think>\n\n"


class TestRenderChatPrompt:
    def test_qwen3_5_nothink_is_a_bare_user_turn(self):
        assert render_chat_prompt("Frame.", "Neutral text.", "qwen3_5_nothink") == USER

    def test_qwen_adds_llamafactory_default_system(self):
        assert render_chat_prompt("Frame.", "Neutral text.", "qwen") == (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            + USER
        )

    @pytest.mark.parametrize("template", ["qwen3_5", "qwen3_6", "qwen3_8"])
    def test_reasoning_template_without_thinking_prefills_empty_thought(self, template):
        # LlamaFactory puts the empty thought in the prompt (no loss) when
        # enable_thinking is false, which is also what the model's own chat
        # template emits with enable_thinking=False.
        assert render_chat_prompt("Frame.", "Neutral text.", template, enable_thinking=False) == USER + EMPTY_THOUGHT

    def test_reasoning_template_with_thinking_leaves_thought_to_the_model(self):
        # With enable_thinking true LlamaFactory trains on the empty thought,
        # so the model writes it itself.
        assert render_chat_prompt("Frame.", "Neutral text.", "qwen3_5", enable_thinking=True) == USER

    def test_qwen3_8_with_thinking_adds_reasoning_effort_system(self):
        prompt = render_chat_prompt("Frame.", "Neutral text.", "qwen3_8", enable_thinking=True)
        assert prompt.startswith("<|im_start|>system\nReasoning effort is set to xhigh.")
        assert prompt.endswith(USER)

    def test_no_instruction_is_just_the_input(self):
        assert render_chat_prompt("", "Neutral text.", "qwen3_5_nothink") == (
            "<|im_start|>user\nNeutral text.<|im_end|>\n<|im_start|>assistant\n"
        )

    def test_persona_in_system_turn_matches_native_template(self):
        # Hemmingway-1's chat_template.jinja with a system message and
        # enable_thinking=False renders exactly this.
        assert render_chat_prompt("Frame.", "Neutral text.", "qwen3_8", enable_thinking=False,
                                  persona_turn="system") == (
            "<|im_start|>system\nFrame.<|im_end|>\n"
            "<|im_start|>user\nNeutral text.<|im_end|>\n<|im_start|>assistant\n" + EMPTY_THOUGHT
        )

    def test_persona_system_replaces_qwen_default_system(self):
        prompt = render_chat_prompt("Frame.", "Neutral text.", "qwen", persona_turn="system")
        assert prompt.startswith("<|im_start|>system\nFrame.<|im_end|>\n")
        assert "You are Qwen" not in prompt

    def test_qwen3_8_thinking_puts_reasoning_before_persona(self):
        prompt = render_chat_prompt("Frame.", "Neutral text.", "qwen3_8", enable_thinking=True,
                                    persona_turn="system")
        assert prompt.startswith(f"<|im_start|>system\n{QWEN3_8_XHIGH_REASONING}\n\nFrame.<|im_end|>\n")

    def test_unknown_persona_turn_is_an_error(self):
        with pytest.raises(ValueError, match="assistant"):
            render_chat_prompt("Frame.", "Neutral text.", "qwen3_8", persona_turn="assistant")

    def test_unknown_template_is_an_error(self):
        with pytest.raises(ValueError, match="vanilla"):
            render_chat_prompt("Frame.", "Neutral text.", "vanilla")


class TestTemplateResolution:
    def _mlx_generator(self, config=None, metadata=None):
        from src.generation.lora_generator import LoRAStyleGenerator
        gen = LoRAStyleGenerator.__new__(LoRAStyleGenerator)
        gen.config = config or GenerationConfig(skip_cleaning=True)
        gen.metadata = metadata
        gen._model = object()
        gen._tokenizer = MagicMock()
        gen._tokenizer.eos_token_ids = set()
        gen._tokenizer.convert_tokens_to_ids.return_value = 7
        gen._logit_bias_processor = False
        gen._ensure_loaded = lambda: None
        return gen

    def _prompt_sent_to_mlx(self, gen):
        # create=True: without MLX installed (CI) the module never imports these.
        with patch("src.generation.lora_generator.generate", return_value="out", create=True) as gen_fn, \
             patch("src.generation.lora_generator.make_sampler", create=True), \
             patch("src.generation.lora_generator.make_repetition_penalty", create=True):
            gen.generate(content="Neutral text.", author="X", instruction="Frame.", raw_prompt=True)
        return gen_fn.call_args.kwargs["prompt"]

    def test_metadata_template_is_used(self):
        from src.generation.lora_generator import AdapterMetadata
        meta = AdapterMetadata(author="X", base_model="m", template="qwen3_8", enable_thinking=False)
        assert self._prompt_sent_to_mlx(self._mlx_generator(metadata=meta)) == USER + EMPTY_THOUGHT

    def test_metadata_persona_turn_is_used(self):
        from src.generation.lora_generator import AdapterMetadata
        meta = AdapterMetadata(author="X", base_model="m", template="qwen3_8", enable_thinking=False,
                               persona_turn="system")
        assert self._prompt_sent_to_mlx(self._mlx_generator(metadata=meta)).startswith(
            "<|im_start|>system\nFrame.<|im_end|>\n<|im_start|>user\nNeutral text.")

    def test_config_overrides_metadata(self):
        from src.generation.lora_generator import AdapterMetadata
        meta = AdapterMetadata(author="X", base_model="m", template="qwen3_8", enable_thinking=False)
        config = GenerationConfig(skip_cleaning=True, chat_template="qwen")
        assert "You are Qwen" in self._prompt_sent_to_mlx(self._mlx_generator(config=config, metadata=meta))

    def test_default_is_qwen3_5_nothink(self):
        assert self._prompt_sent_to_mlx(self._mlx_generator()) == USER

    def test_im_end_stops_generation(self):
        gen = self._mlx_generator()
        self._prompt_sent_to_mlx(gen)
        assert 7 in gen._tokenizer.eos_token_ids

    def test_metadata_reads_template_fields(self, tmp_path):
        from src.generation.lora_generator import AdapterMetadata
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps({"author": "A", "base_model": "m", "template": "qwen3_8",
                                    "enable_thinking": False, "persona_turn": "system"}))
        meta = AdapterMetadata.from_file(path)
        assert (meta.template, meta.enable_thinking, meta.persona_turn) == ("qwen3_8", False, "system")

    def test_adapter_config_fields_reach_generation_config(self):
        from src.config import _parse_model_config
        model = _parse_model_config({"chat_template": "qwen", "enable_thinking": False, "persona_turn": "system"})
        assert (model.chat_template, model.enable_thinking, model.persona_turn) == ("qwen", False, "system")
        with patch("src.config.get_adapter_config", return_value=model):
            config = GenerationConfig.from_config("lora_adapters/x")
        assert (config.chat_template, config.enable_thinking, config.persona_turn) == ("qwen", False, "system")


class TestPyTorchGenerator:
    def test_uses_training_template_and_stops_at_im_end(self):
        torch = pytest.importorskip("torch")
        from src.generation.lora_generator import AdapterMetadata
        from src.generation.pytorch_generator import PyTorchStyleGenerator

        gen = PyTorchStyleGenerator.__new__(PyTorchStyleGenerator)
        gen.config = GenerationConfig(skip_cleaning=True)
        gen.metadata = AdapterMetadata(author="X", base_model="m", template="qwen3_8", enable_thinking=False)
        gen._ensure_loaded = lambda: None
        gen._tokenizer = MagicMock()
        gen._tokenizer.eos_token_id = 1
        gen._tokenizer.convert_tokens_to_ids.return_value = 7
        gen._tokenizer.return_value.to.return_value = {"input_ids": torch.zeros((1, 3), dtype=torch.long)}
        gen._tokenizer.decode.return_value = "out"
        gen._model = MagicMock()
        gen._model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)

        gen.generate(content="Neutral text.", author="X", instruction="Frame.", raw_prompt=True)

        assert gen._tokenizer.call_args.args[0] == USER + EMPTY_THOUGHT
        gen._tokenizer.apply_chat_template.assert_not_called()
        assert 7 in gen._model.generate.call_args.kwargs["eos_token_id"]

    def test_pytorch_metadata_reads_converter_metadata(self, tmp_path):
        from src.generation.pytorch_generator import PyTorchAdapterMetadata
        (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "m", "r": 8}))
        (tmp_path / "metadata.json").write_text(json.dumps({"template": "qwen", "enable_thinking": False,
                                                            "persona_turn": "system"}))
        meta = PyTorchAdapterMetadata.from_adapter_config(str(tmp_path))
        assert (meta.template, meta.enable_thinking, meta.persona_turn, meta.lora_rank) == ("qwen", False, "system", 8)
