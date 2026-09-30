"""Tests for semantic fidelity validation."""

import json
import pytest
from unittest.mock import MagicMock


class TestFidelityResult:
    """Tests for FidelityResult dataclass."""

    def test_was_modified_true_when_changes(self):
        from src.validation.semantic_fidelity import FidelityResult

        result = FidelityResult(
            original="original",
            corrected="corrected",
            changes=[{"issue": "test", "fix": "fixed"}],
        )
        assert result.was_modified is True

    def test_was_modified_false_when_no_changes(self):
        from src.validation.semantic_fidelity import FidelityResult

        result = FidelityResult(
            original="original",
            corrected="original",
            changes=[],
        )
        assert result.was_modified is False

    def test_default_changes_empty(self):
        from src.validation.semantic_fidelity import FidelityResult

        result = FidelityResult(original="a", corrected="a")
        assert result.changes == []
        assert result.was_modified is False


class TestValidateSemanticFidelity:
    """Tests for validate_semantic_fidelity function."""

    def test_no_changes_needed(self):
        """When restyled text is faithful, return it unchanged."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "The cat sat on the mat.",
        })

        result = validate_semantic_fidelity(
            original="The cat sat on the mat.",
            restyled="The cat sat on the mat.",
            critic_provider=mock_provider,
        )

        assert result.corrected == "The cat sat on the mat."
        assert result.was_modified is False
        assert result.changes == []

    def test_corrects_factual_error(self):
        """When restyled has a factual error, return corrected version."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"issue": "wrong subject", "fix": "fixed subject"}],
            "result": "The dog sat on the mat.",
        })

        result = validate_semantic_fidelity(
            original="The dog sat on the mat.",
            restyled="The cat sat on the mat.",
            critic_provider=mock_provider,
        )

        assert result.corrected == "The dog sat on the mat."
        assert result.was_modified is True
        assert len(result.changes) == 1

    def test_passes_correct_prompts(self):
        """Verify the function sends original and restyled in the user prompt."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "restyled text",
        })

        validate_semantic_fidelity(
            original="original text",
            restyled="restyled text",
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")
        assert "ORIGINAL:" in user_prompt
        assert "original text" in user_prompt
        assert "RESTYLED:" in user_prompt
        assert "restyled text" in user_prompt

    def test_uses_low_temperature(self):
        """Validation should use low temperature for deterministic output."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "text",
        })

        validate_semantic_fidelity(
            original="text",
            restyled="text",
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.1

    def test_requests_json_format(self):
        """Validation should request JSON response format."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "text",
        })

        validate_semantic_fidelity(
            original="text",
            restyled="text",
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        assert call_kwargs.kwargs.get("require_json") is True

    def test_fallback_on_json_parse_error(self):
        """If LLM returns invalid JSON, return restyled text unchanged."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = "This is not JSON at all"

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled unchanged",
            critic_provider=mock_provider,
        )

        assert result.corrected == "restyled unchanged"
        assert result.was_modified is False

    def test_fallback_on_api_error(self):
        """If LLM call fails, return restyled text unchanged."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.side_effect = RuntimeError("API timeout")

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled unchanged",
            critic_provider=mock_provider,
        )

        assert result.corrected == "restyled unchanged"
        assert result.was_modified is False

    def test_fallback_on_missing_result_key(self):
        """If JSON is missing 'result' key, use restyled text."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"issue": "something"}],
        })

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled fallback",
            critic_provider=mock_provider,
        )

        # Should use restyled as fallback since "result" key missing
        assert result.corrected == "restyled fallback"

    def test_max_tokens_covers_the_whole_response(self):
        """The reply walks every claim, quotes spans and repeats the paragraph.

        With max_tokens at 4x the restyled words, 7 of 15 replies on a
        170-word essay stopped at 1024 tokens mid-JSON and went unchecked.
        """
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({"changes": [], "result": "short"})
        text = " ".join(["word"] * 170)
        validate_semantic_fidelity(original=text, restyled=text, critic_provider=mock_provider)
        budget = mock_provider.call.call_args.kwargs.get("max_tokens")
        assert budget >= 4096
        # Coverage claims ~ the original, found spans + result ~ twice the restyled text.
        long_text = " ".join(["word"] * 900)
        validate_semantic_fidelity(original=long_text, restyled=long_text, critic_provider=mock_provider)
        assert mock_provider.call.call_args.kwargs.get("max_tokens") >= 900 * 3 * 3

    def test_a_cut_off_reply_is_retried_with_more_room(self):
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.side_effect = [
            '{"coverage": [{"claim": "cut off',
            json.dumps({"changes": [{"type": "missing", "issue": "x"}], "result": "fixed text"}),
        ]
        result = validate_semantic_fidelity(original="orig", restyled="restyled", critic_provider=mock_provider)
        assert result.corrected == "fixed text"
        first, second = [c.kwargs["max_tokens"] for c in mock_provider.call.call_args_list]
        assert second > first

    def test_multiple_changes_logged(self):
        """Multiple changes should all be present in result."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        changes = [
            {"issue": "missing claim A", "fix": "added A"},
            {"issue": "reversed meaning B", "fix": "fixed B"},
        ]
        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": changes,
            "result": "corrected text with A and B",
        })

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled",
            critic_provider=mock_provider,
        )

        assert len(result.changes) == 2
        assert result.was_modified is True
        assert result.corrected == "corrected text with A and B"


    def test_rejects_placeholder_result(self):
        """A critic that echoes the schema placeholder must not wipe the paragraph."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        restyled = " ".join(["word"] * 90)
        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "restyled text unchanged",
        })

        result = validate_semantic_fidelity(
            original="original " * 90,
            restyled=restyled,
            critic_provider=mock_provider,
        )

        assert result.corrected == restyled
        assert result.was_modified is False

    def test_rejects_truncated_result(self):
        """A result far shorter than the input is malformed, not a valid edit."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        restyled = " ".join(["word"] * 100)
        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"type": "missing", "issue": "x", "fix": "y"}],
            "result": " ".join(["word"] * 20),
        })

        result = validate_semantic_fidelity(
            original="original",
            restyled=restyled,
            critic_provider=mock_provider,
        )

        assert result.corrected == restyled

    def test_accepts_legitimate_large_cut(self):
        """Cutting a hallucinated sentence is legitimate and must survive."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        restyled = " ".join(["word"] * 100)
        kept = " ".join(["word"] * 65)
        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"type": "contradiction", "issue": "hallucination", "fix": "cut"}],
            "result": kept,
        })

        result = validate_semantic_fidelity(
            original="original",
            restyled=restyled,
            critic_provider=mock_provider,
        )

        assert result.corrected == kept


class TestSemanticFidelityPrompt:
    """Tests for the semantic fidelity prompt file.

    The prompt is the whole behaviour of this check, so these assert the
    properties an A/B run showed to matter, not incidental wording.
    """

    def test_prompt_file_exists(self):
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert len(prompt) > 0

    def test_prompt_describes_json_format(self):
        """Prompt should specify JSON output format."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert '"changes"' in prompt
        assert '"result"' in prompt
        assert '"type"' in prompt

    def test_prompt_has_no_literal_placeholder_result(self):
        """A bracketed placeholder gets echoed verbatim and wipes the paragraph."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert "<restyled text unchanged>" not in prompt
        assert "word for word" in prompt

    def test_prompt_judges_at_paragraph_level(self):
        """Restyling moves claims between sentences; the check must allow that."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity").lower()
        assert "sentence-by-sentence correspondence" in prompt
        assert "whole paragraph" in prompt

    def test_prompt_puts_structure_first(self):
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity").lower()
        assert "structure outranks everything" in prompt
        assert "toward the original's wording" in prompt
        assert "smallest" in prompt

    def test_prompt_requires_in_place_repair(self):
        """Missing info gets folded into an existing sentence, never appended."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert '"host"' in prompt
        assert "NEVER adds a sentence" in prompt

    def test_prompt_keeps_flourish_but_cuts_added_claims(self):
        """Imagery and asides survive; invented claims don't.

        The run-3 Russell adapter added a definition of memory chips, an
        argument about "no further winners" and a quote from "one analyst at
        a top-tier Wall Street firm"; the old rule let all three through.
        """
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        lower = prompt.lower()
        assert "imagery" in lower and "aside" in lower
        for kind in ("claim", "explanation", "example", "attribution", "quotation"):
            assert kind in lower, kind
        assert '"added"' in prompt
        assert "added material is allowed" not in lower

    def test_prompt_requires_both_walks(self):
        """Walk 1 catches drops and distortions, Walk 2 catches invented specifics."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert '"coverage"' in prompt
        assert '"specifics"' in prompt
        assert "faithful" in prompt
        assert "in_original" in prompt

    def test_prompt_blocks_invented_attribution_and_counterparty(self):
        """The two specifics an A/B run showed slip through most easily."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert "ATTRIBUTION" in prompt
        assert "COUNTERPARTY" in prompt
        assert "never more specific than it" in prompt

    def test_prompt_protects_deliberate_fragments(self):
        """Stylistic fragments are not grammar errors."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity").lower()
        assert "fragments" in prompt
        assert "style, not errors" in prompt

    def test_prompt_requires_final_reread(self):
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity").lower()
        assert "final check" in prompt


class TestChangeLogging:
    """Change classification should surface in logs."""

    def test_logs_change_type_when_present(self, caplog):
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        restyled = " ".join(["word"] * 40) + " and a storm"
        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"type": "added", "issue": "invented a storm", "fix": "cut clause"}],
            "result": " ".join(["word"] * 40),
        })

        with caplog.at_level("INFO"):
            validate_semantic_fidelity(
                original=" ".join(["word"] * 40),
                restyled=restyled,
                critic_provider=mock_provider,
            )

        assert "added" in caplog.text
        assert "invented a storm" in caplog.text

    def test_logs_without_type(self, caplog):
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        text = " ".join(["word"] * 40)
        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"issue": "dropped a claim", "fix": "restored"}],
            "result": text,
        })

        with caplog.at_level("INFO"):
            validate_semantic_fidelity(
                original=text,
                restyled=text,
                critic_provider=mock_provider,
            )

        assert "dropped a claim" in caplog.text


class TestTransferPipelineIntegration:
    """Tests for semantic fidelity integration in the transfer pipeline."""

    def test_verify_semantic_fidelity_config_default(self):
        """verify_semantic_fidelity should default to True."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()
        assert config.verify_semantic_fidelity is True

    def test_verify_semantic_fidelity_can_disable(self):
        """verify_semantic_fidelity can be set to False."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(verify_semantic_fidelity=False)
        assert config.verify_semantic_fidelity is False

    def test_cli_no_verify_survives_adapter_config(self):
        """--no-verify must not be clobbered by an adapter's verify_entailment=true."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(
            verify_semantic_fidelity=False,
            verify_semantic_fidelity_explicit=True,
        )
        assert config.verify_semantic_fidelity is False
        assert config.verify_semantic_fidelity_explicit is True

    def test_verify_explicit_defaults_false(self):
        """Without the CLI flag, config-driven verification still applies."""
        from src.generation.transfer import TransferConfig

        assert TransferConfig().verify_semantic_fidelity_explicit is False



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
