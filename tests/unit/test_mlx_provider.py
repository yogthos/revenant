"""Tests for mlx_provider module.

Tests cover:
- Bug 8: Infinite recursion in _neutralize_chunked for long single sentences
- Parity: MLX and DeepSeek neutralizers share _extract_entities / _restore_entities
  / _monotone_flatten byte-for-byte. The parity tests pin that equivalence so a
  later Round 2 refactor (extracting a BaseRTTNeutralizer) is a no-op on behavior.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestNeutralizeChunked:
    """Tests for _neutralize_chunked infinite recursion (Bug 8)."""

    def test_long_single_sentence_no_recursion(self):
        """A 400-word sentence with no periods should not recurse infinitely."""
        from src.llm.mlx_provider import RTTNeutralizer

        # Create a mock instance
        provider = RTTNeutralizer.__new__(RTTNeutralizer)
        provider._model = None
        provider._tokenizer = None

        # 400-word sentence with no sentence-ending punctuation
        long_sentence = " ".join(["word"] * 400)

        call_count = 0
        original_neutralize = provider.neutralize.__func__ if hasattr(provider.neutralize, '__func__') else None

        def mock_neutralize(text, max_retries=2, monotone=False):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise RecursionError("Infinite recursion detected!")
            # For chunks that are still >300 words, should go to _do_neutralize
            if len(text.split()) > 300:
                return provider._neutralize_chunked(text, max_retries, monotone)
            return f"neutralized: {text[:50]}"

        def mock_do_neutralize(text, max_retries=2, monotone=False):
            return f"directly neutralized: {text[:50]}"

        provider.neutralize = mock_neutralize
        provider._do_neutralize = mock_do_neutralize

        # Should not raise RecursionError
        result = provider._neutralize_chunked(long_sentence, max_retries=2, monotone=False)
        assert result is not None
        assert call_count <= 5  # Should not recurse excessively

    def test_chunk_over_300_words_handled(self):
        """Chunks >300 words should go to _do_neutralize directly."""
        from src.llm.mlx_provider import RTTNeutralizer

        provider = RTTNeutralizer.__new__(RTTNeutralizer)
        provider._model = None
        provider._tokenizer = None

        # Text that produces a chunk >300 words (no sentence boundaries)
        long_text = " ".join(["word"] * 400)

        do_neutralize_called = False

        def mock_do_neutralize(text, max_retries=2, monotone=False):
            nonlocal do_neutralize_called
            do_neutralize_called = True
            return "neutralized text"

        provider._do_neutralize = mock_do_neutralize

        result = provider._neutralize_chunked(long_text, max_retries=2, monotone=False)
        assert do_neutralize_called, "_do_neutralize should be called for chunks >300 words"
        assert result is not None


class TestRttOnceHelper:
    """_rtt_once: single pass through Mandarin→English, shared by neutralize()
    and _do_neutralize() to eliminate the ~60-line duplicated loop body."""

    def _provider(self):
        from src.llm.mlx_provider import RTTNeutralizer
        provider = RTTNeutralizer.__new__(RTTNeutralizer)
        provider._model = None
        provider._tokenizer = None
        return provider

    def test_empty_mandarin_returns_none(self):
        provider = self._provider()
        provider._generate = MagicMock(side_effect=["", "unused"])

        result = provider._rtt_once("some English text here.", word_count=4)
        assert result is None

    def test_empty_english_returns_none(self):
        provider = self._provider()
        provider._generate = MagicMock(side_effect=["mandarin output", ""])

        result = provider._rtt_once("some English text here.", word_count=4)
        assert result is None

    def test_chinese_residue_returns_none(self):
        """If the Mandarin→English step leaks Chinese characters, pass failed."""
        provider = self._provider()
        provider._generate = MagicMock(
            side_effect=["some mandarin", "mixed with 中文 characters here"]
        )

        result = provider._rtt_once("source text.", word_count=3)
        assert result is None

    def test_strips_code_fences(self):
        """Leading/trailing ``` fences should be stripped from the output."""
        provider = self._provider()
        provider._generate = MagicMock(
            side_effect=["some mandarin text", "```\nplain english output\n```"]
        )

        result = provider._rtt_once("source.", word_count=2)
        assert result == "plain english output"

    def test_success_returns_cleaned_english(self):
        provider = self._provider()
        provider._generate = MagicMock(
            side_effect=["mandarin translation", "  clean english output  "]
        )

        result = provider._rtt_once("source here.", word_count=3)
        assert result == "clean english output"


class TestNeutralizerSharedMethodParity:
    """Pin that MLX and DeepSeek neutralizers agree on the pure helper methods.

    This is the Round 1 safety net for Round 2's BaseRTTNeutralizer extraction:
    the three methods below (_extract_entities, _restore_entities, _monotone_flatten)
    are currently duplicated verbatim across the two classes. After extraction,
    both subclasses must still produce identical output for the same input — these
    tests fail loudly if the refactor drifts.
    """

    @pytest.fixture
    def mlx_neutralizer(self):
        from src.llm.mlx_provider import RTTNeutralizer
        obj = RTTNeutralizer.__new__(RTTNeutralizer)
        obj._model = None
        obj._tokenizer = None
        return obj

    @pytest.fixture
    def ds_neutralizer(self):
        from src.llm.mlx_provider import DeepSeekRTTNeutralizer
        obj = DeepSeekRTTNeutralizer.__new__(DeepSeekRTTNeutralizer)
        # Don't call __init__ — avoids the API key requirement.
        return obj

    # Samples exercise: multi-word names, single caps, sentence-start caps,
    # embedded punctuation, short + long inputs, already-placeholdered text.
    SAMPLES = [
        "Jervas Dudley walked through New England toward Squire Brewster Hyde.",
        "The shadows lengthened as Cthulhu stirred in R'lyeh beneath the waves.",
        "However, although Paris fell in June, the resistance endured.",
        "Short text.",
        "Already __ENT0__ masked text with Paris inside.",
        "Punctuation: commas, semicolons; dashes — and (parentheticals).",
    ]

    @pytest.mark.parametrize("text", SAMPLES)
    def test_extract_entities_parity(self, mlx_neutralizer, ds_neutralizer, text):
        mlx_masked, mlx_map = mlx_neutralizer._extract_entities(text)
        ds_masked, ds_map = ds_neutralizer._extract_entities(text)
        assert mlx_masked == ds_masked
        assert mlx_map == ds_map

    @pytest.mark.parametrize("text", SAMPLES)
    def test_restore_entities_parity(self, mlx_neutralizer, ds_neutralizer, text):
        masked, entity_map = mlx_neutralizer._extract_entities(text)
        assert mlx_neutralizer._restore_entities(masked, entity_map) == \
            ds_neutralizer._restore_entities(masked, entity_map)

    MONOTONE_SAMPLES = [
        "The old man walked slowly down the lane. He saw many things.",
        "She came in, she sat down (quietly), she smiled; the room brightened.",
        "A very long sentence that rambles and meanders and connects many clauses with conjunctions and never quite ends properly.",
        "Short. Fragments. Here.",
        "No punctuation at all just words trailing",
    ]

    @pytest.mark.parametrize("text", MONOTONE_SAMPLES)
    def test_monotone_flatten_parity(self, mlx_neutralizer, ds_neutralizer, text):
        assert mlx_neutralizer._monotone_flatten(text) == \
            ds_neutralizer._monotone_flatten(text)

    def test_extract_entities_round_trip(self, mlx_neutralizer):
        """Masking then restoring must recover the original text exactly."""
        text = "Jervas Dudley walked through New England at dawn."
        masked, entity_map = mlx_neutralizer._extract_entities(text)
        assert mlx_neutralizer._restore_entities(masked, entity_map) == text

    def test_extract_entities_skips_sentence_start_single_caps(self, mlx_neutralizer):
        """Single capitalized word at sentence start is not masked (pattern 2 requires
        preceding punctuation/whitespace)."""
        masked, entity_map = mlx_neutralizer._extract_entities("The door opened.")
        assert "__ENT" not in masked
        assert entity_map == {}


class TestDeadActiveWorkersRemoved:
    """Regression guard for the dead `active_workers`/`workers_lock` counter that
    was written but never read. Removing them lets the ThreadPoolExecutor join
    naturally and cuts one class of future drift (a reader adding logic that
    relies on the stale counter)."""

    def test_no_active_workers_counter(self):
        import inspect
        from src.llm import mlx_provider

        source = inspect.getsource(mlx_provider)
        assert "active_workers" not in source, (
            "active_workers counter was dead code — never read. "
            "ThreadPoolExecutor.futures already tracks completion."
        )
        assert "workers_lock" not in source, (
            "workers_lock guarded only the dead active_workers counter."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Training-data fidelity fixes
# =============================================================================

@pytest.fixture
def base_neutralizer():
    from src.llm.mlx_provider import DeepSeekRTTNeutralizer
    return DeepSeekRTTNeutralizer.__new__(DeepSeekRTTNeutralizer)


class TestMonotoneFlattenKeepsContent:
    """Flattening must never delete words from the text."""

    @staticmethod
    def _words(text):
        import re
        return sorted(re.findall(r"[a-z0-9']+", text.lower()))

    def test_keeps_text_between_dashes(self, base_neutralizer):
        text = "Men fear thought — more than ruin — and they fear it more than death."
        out = base_neutralizer._monotone_flatten(text)
        assert "more than ruin" in out
        assert self._words(out) == self._words(text)

    def test_dashes_across_sentences_keep_both_sentences(self, base_neutralizer):
        text = "The war began — slowly. Then the peace came — and nobody noticed."
        out = base_neutralizer._monotone_flatten(text)
        assert "slowly" in out and "Then the peace came" in out
        assert self._words(out) == self._words(text)

    def test_keeps_parentheticals(self, base_neutralizer):
        text = "The book (published in 1918) was banned in several countries."
        out = base_neutralizer._monotone_flatten(text)
        assert "published in 1918" in out
        assert self._words(out) == self._words(text)

    def test_keeps_short_clause_fragments(self, base_neutralizer):
        text = ("He studied logic and mathematics for many years at Cambridge, but failed, "
                "and he then turned his attention toward politics and the war.")
        out = base_neutralizer._monotone_flatten(text)
        assert "failed" in out
        conjunctions = {"and", "but", "or", "yet", "so", "however", "although", "while", "whereas"}
        assert [w for w in self._words(out) if w not in conjunctions] == \
            [w for w in self._words(text) if w not in conjunctions]


class TestEntityMaskingSentenceStart:
    def test_sentence_initial_word_not_masked(self, base_neutralizer):
        masked, entity_map = base_neutralizer._extract_entities(
            "Religion is fear. Nothing else explains its hold on the mind."
        )
        assert "Nothing" not in entity_map.values()
        assert "Religion" not in entity_map.values()
        assert "__ENT" not in masked

    def test_sentence_initial_after_quote_or_newline_not_masked(self, base_neutralizer):
        masked, entity_map = base_neutralizer._extract_entities(
            'He said "Nothing matters." Philosophers\ndisagree.\nCertainly not.'
        )
        assert "Nothing" not in entity_map.values()
        assert "Certainly" not in entity_map.values()

    def test_mid_sentence_name_still_masked(self, base_neutralizer):
        masked, entity_map = base_neutralizer._extract_entities(
            "The argument that Wittgenstein made was obscure."
        )
        assert "Wittgenstein" in entity_map.values()

    def test_sentence_initial_multiword_name_still_masked(self, base_neutralizer):
        masked, entity_map = base_neutralizer._extract_entities(
            "It rained. Alfred Whitehead arrived late."
        )
        assert "Alfred Whitehead" in entity_map.values()


class TestPlaceholderRecovery:
    def test_restores_mangled_placeholders(self, base_neutralizer):
        entity_map = {"__ENT0__": "Wittgenstein", "__ENT1__": "Cambridge"}
        text = "__ENT0_ went to ENT1 and later __ent0__ left."
        out = base_neutralizer._restore_entities(text, entity_map)
        assert out == "Wittgenstein went to Cambridge and later Wittgenstein left."

    def test_has_placeholder_residue(self, base_neutralizer):
        from src.llm.mlx_provider import has_placeholder_residue
        assert has_placeholder_residue("He met __ENT3__ there.")
        assert has_placeholder_residue("He met ENT3 there.")
        assert not has_placeholder_residue("He met Kant there.")


class TestBatchResponseParser:
    def test_line_starting_with_year_is_not_an_item_marker(self):
        from src.llm.mlx_provider import parse_numbered_response
        response = "[1] The war ended.\n1918 was a hard year for everyone.\n[2] Peace came slowly."
        items = parse_numbered_response(response, expected=2)
        assert items[1] == "The war ended. 1918 was a hard year for everyone."
        assert items[2] == "Peace came slowly."

    def test_no_dot_prefix_on_numbered_items(self):
        from src.llm.mlx_provider import parse_numbered_response
        items = parse_numbered_response("1. First text here.\n2. Second text here.", expected=2)
        assert items == {1: "First text here.", 2: "Second text here."}

    def test_out_of_range_numbers_are_text(self):
        from src.llm.mlx_provider import parse_numbered_response
        items = parse_numbered_response("[1] Alpha.\n3. Beta continues.", expected=2)
        assert items == {1: "Alpha. 3. Beta continues."}

    def test_bracketed_markers_preferred_when_present(self):
        from src.llm.mlx_provider import parse_numbered_response
        response = "[1] Points:\n2. not a marker\n[2] Second."
        items = parse_numbered_response(response, expected=2)
        assert items == {1: "Points: 2. not a marker", 2: "Second."}


class TestBatchTruncationAndLength:
    def _neutralizer(self, response, finish_reason="stop"):
        from src.llm.mlx_provider import DeepSeekRTTNeutralizer
        obj = DeepSeekRTTNeutralizer.__new__(DeepSeekRTTNeutralizer)
        obj.temperature = 0.1
        obj.max_tokens = 4000
        obj._call_api_full = MagicMock(return_value=(response, finish_reason))
        return obj

    TEXTS = [
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa.",
        "One two three four five six seven eight nine ten eleven twelve.",
    ]

    REWORDED = "Kappa iota theta eta zeta, epsilon delta gamma beta alpha."

    def test_truncated_response_drops_last_item(self):
        response = (f"[1] {self.REWORDED}\n"
                    "[2] One two three four")
        obj = self._neutralizer(response, finish_reason="length")
        results = dict(obj._process_single_batch((0, self.TEXTS)))
        assert 0 in results
        assert 1 not in results

    def test_too_short_item_rejected(self):
        response = (f"[1] {self.REWORDED}\n"
                    "[2] One two.")
        obj = self._neutralizer(response)
        results = dict(obj._process_single_batch((0, self.TEXTS)))
        assert 0 in results
        assert 1 not in results

    def test_unrestorable_placeholder_rejected(self):
        texts = ["The idea that Wittgenstein held was that language pictures the world as facts."]
        response = "[1] The idea __ENT9__ held was that language pictures the world as facts."
        obj = self._neutralizer(response)
        results = dict(obj._process_single_batch((0, texts)))
        assert results == {}

    def test_echoed_item_is_rejected(self):
        # DeepSeek sometimes copies later items of a batch verbatim; an echo
        # is not neutral input and has to be retried.
        texts = ["He then proceeds to consider common objects, such as a tree, and he shows that all we know "
                 "immediately when we perceive the tree consists of ideas in his sense of the word."]
        obj = self._neutralizer(f"[1] {texts[0]}")
        assert obj._process_single_batch((0, texts)) == []

    def test_reworded_item_with_shared_nouns_is_kept(self):
        texts = ["He then proceeds to consider common objects, such as a tree, and he shows that all we know "
                 "immediately when we perceive the tree consists of ideas in his sense of the word."]
        reworded = ("Next he looks at everyday things like a tree. He argues that what we directly know when we "
                    "see the tree is made of ideas, in the meaning he gives that word.")
        obj = self._neutralizer(f"[1] {reworded}")
        assert dict(obj._process_single_batch((0, texts))) == {0: reworded}

    def test_api_error_does_not_recurse(self):
        from src.llm.mlx_provider import DeepSeekRTTNeutralizer
        obj = DeepSeekRTTNeutralizer.__new__(DeepSeekRTTNeutralizer)
        obj._call_api_full = MagicMock(side_effect=RuntimeError("boom"))
        assert obj._process_single_batch((0, self.TEXTS)) == []

    def test_single_neutralize_uses_batch_prompt(self):
        response = f"[1] {self.REWORDED}"
        obj = self._neutralizer(response)
        out = obj.neutralize(self.TEXTS[0], monotone=False)
        assert out == self.REWORDED
        system = obj._call_api_full.call_args.kwargs["system"]
        from src.utils.prompts import load_prompt
        assert system == load_prompt("rtt_deepseek_batch")


class TestNeutralizerPromptsDontAddStyle:
    @pytest.mark.parametrize("name", ["rtt_deepseek_batch"])
    def test_no_style_instructions(self, name):
        from src.utils.prompts import load_prompt
        text = load_prompt(name).lower()
        assert "vary sentence length" not in text
        assert "use contractions" not in text
        assert "hsk 5" in text or "hsk5" in text  # matches the documented pipeline

    def test_asks_for_rewording(self):
        from src.utils.prompts import load_prompt
        text = load_prompt("rtt_deepseek_batch").lower()
        assert "never copy a run of four or more words" in text


class TestChunkedKeepsOuterPlaceholders:
    def test_chunks_with_outer_placeholders_are_not_rejected(self):
        from src.llm.mlx_provider import RTTNeutralizer
        obj = RTTNeutralizer.__new__(RTTNeutralizer)
        obj._model = None
        obj._tokenizer = None
        # Echo the chunk back: outer placeholders survive until the caller restores them.
        obj._rtt_once = MagicMock(side_effect=lambda text, wc: text)
        sentence = "The philosopher Wittgenstein argued at length about the limits of language and thought. "
        text = sentence * 30  # > 300 words forces chunking
        out = obj.neutralize(text, max_retries=1)
        assert out is not None
        assert "Wittgenstein" in out
        assert "__ENT" not in out
