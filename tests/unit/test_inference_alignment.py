"""Inference must present inputs the way LlamaFactory presented them in training."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestChatLayout:
    def test_instruction_and_input_share_one_user_turn(self):
        from src.generation.base_generator import chat_messages
        assert chat_messages("Persona frame.\n\nWrite approximately 50 words.", "Neutral text.") == [
            {"role": "user", "content": "Persona frame.\n\nWrite approximately 50 words.\nNeutral text."}
        ]

    def test_no_instruction_is_just_the_input(self):
        from src.generation.base_generator import chat_messages
        assert chat_messages("", "Neutral text.") == [{"role": "user", "content": "Neutral text."}]

    def test_multi_paragraph_input_stays_in_user_turn(self):
        from src.generation.base_generator import split_legacy_prompt
        prompt = "Frame.\n\nWrite approximately 9 words.\n\nFirst para.\n\nSecond para.\n###"
        instruction, content = split_legacy_prompt(prompt, content="First para.\n\nSecond para.")
        assert instruction == "Frame.\n\nWrite approximately 9 words."
        assert content == "First para.\n\nSecond para."


class TestPersonaInstruction:
    @patch("src.persona.prompt_builder._get_persona_frame", return_value="FRAME")
    @patch("src.persona.prompt_builder._detect_content_type", return_value=False)
    def test_instruction_excludes_content_and_stop_marker(self, _detect, _frame):
        from src.persona.prompt_builder import build_persona_instruction
        instruction = build_persona_instruction("Neutral text here.", target_words=40)
        assert instruction.startswith("FRAME\n\nWrite approximately 40 words.")
        assert "Neutral text here." not in instruction
        assert "###" not in instruction

    @patch("src.persona.prompt_builder._get_persona_frame", return_value="FRAME")
    @patch("src.persona.prompt_builder._detect_content_type", return_value=False)
    def test_classifies_the_input(self, detect, _frame):
        from src.persona.prompt_builder import build_persona_instruction
        build_persona_instruction("Neutral text here.", target_words=40)
        assert detect.call_args.args[0] == "Neutral text here."


    def test_instruction_has_only_what_training_rows_had(self):
        # Training never saw RAG rhythm hints or LLM-extracted skeletons of
        # another paragraph, so the instruction builder takes neither.
        import inspect
        from src.persona.prompt_builder import build_persona_instruction, build_persona_prompt
        for fn in (build_persona_instruction, build_persona_prompt):
            params = inspect.signature(fn).parameters
            assert "structural_guidance" not in params
            assert "grafting_guidance" not in params


class TestLengthHint:
    def test_default_expansion_matches_training_ratio(self):
        from src.generation.transfer import TransferConfig
        from src.config import GenerationConfig
        assert TransferConfig().target_expansion_ratio == pytest.approx(1.25)
        assert GenerationConfig().target_expansion_ratio == pytest.approx(1.25)

    def test_sample_config_uses_training_ratio(self):
        root = Path(__file__).parent.parent.parent
        config = json.loads((root / "config.json.sample").read_text())
        assert config["generation"]["target_expansion_ratio"] == pytest.approx(1.25)


class TestInferenceNeutralization:
    @patch("src.generation.transfer.create_style_generator")
    def test_rtt_uses_monotone_flattening_like_training(self, _gen):
        from src.generation.transfer import StyleTransfer, TransferConfig
        transfer = StyleTransfer(
            adapter_path=None, author_name="Test", critic_provider=MagicMock(provider_name="mock"),
            config=TransferConfig(verify_semantic_fidelity=False, use_structural_rag=False,
                                  use_structural_grafting=False),
        )
        transfer._rtt_neutralizer = MagicMock()
        transfer._rtt_neutralizer.neutralize.return_value = "plain"
        transfer._rtt_neutralize("Some text.")
        assert transfer._rtt_neutralizer.neutralize.call_args.kwargs["monotone"] is True

    @patch("src.generation.transfer.create_style_generator")
    def test_persona_path_passes_instruction_separately(self, mock_gen_factory):
        from src.generation.transfer import StyleTransfer, TransferConfig
        generator = MagicMock()
        generator.generate.return_value = "Styled output text from the generator model here."
        mock_gen_factory.return_value = generator
        transfer = StyleTransfer(
            adapter_path=None, author_name="Test", critic_provider=MagicMock(provider_name="mock"),
            config=TransferConfig(verify_semantic_fidelity=False, skip_neutralization=True,
                                  use_persona=True, apply_input_perturbation=False,
                                  use_structural_rag=False, use_structural_grafting=False,
                                  min_paragraph_words=3),
        )
        with patch("src.generation.transfer.build_persona_instruction", return_value="INSTRUCTION"):
            transfer.transfer_paragraph("The first line of this paragraph has several words in it.")
        kwargs = generator.generate.call_args.kwargs
        assert kwargs["instruction"] == "INSTRUCTION"
        assert kwargs["content"] == "The first line of this paragraph has several words in it."

    @patch("src.generation.transfer.create_style_generator")
    def test_persona_path_skips_rag_and_grafting_lookups(self, mock_gen_factory):
        from src.generation.transfer import StyleTransfer, TransferConfig
        generator = MagicMock()
        generator.generate.return_value = "Styled output text from the generator model here."
        mock_gen_factory.return_value = generator
        transfer = StyleTransfer(
            adapter_path=None, author_name="Test", critic_provider=MagicMock(provider_name="mock"),
            config=TransferConfig(verify_semantic_fidelity=False, skip_neutralization=True,
                                  use_persona=True, apply_input_perturbation=False,
                                  use_structural_rag=False, use_structural_grafting=False,
                                  min_paragraph_words=3),
        )
        transfer.structural_rag = MagicMock()
        transfer.structural_grafter = MagicMock()
        with patch("src.generation.transfer.build_persona_instruction", return_value="INSTRUCTION"):
            transfer.transfer_paragraph("The first line of this paragraph has several words in it.")
        transfer.structural_rag.get_guidance.assert_not_called()
        transfer.structural_grafter.get_grafting_guidance.assert_not_called()
