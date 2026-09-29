"""LLM-style inputs: teach the model to rebuild the author's structure.

Retold inputs keep the author's sentence order and argument, so the model
only learned to swap vocabulary; given LLM-structured text it kept the LLM
structure. llm_style rows start from a rewrite of the author's paragraph in
typical LLM prose (thesis first, signposts, reversals, tricolons, summary
closers), run through the same RTT as inference, with the author's paragraph
as the target.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

RUSSELL = (
    "The man who has no tincture of philosophy goes through life imprisoned in the prejudices derived "
    "from common sense, from the habitual beliefs of his age or his nation, and from convictions which "
    "have grown up in his mind without the co-operation or consent of his deliberate reason. To such a "
    "man the world tends to become definite, finite, obvious; common objects rouse no questions, and "
    "unfamiliar possibilities are contemptuously rejected."
)
REWRITE = (
    "Without philosophy, a person stays trapped. Common sense, the beliefs of their era and nation, and "
    "ideas they never consciously chose all shape how they think. The result? The world feels fixed, "
    "limited, and obvious. Everyday things spark no curiosity. New possibilities get dismissed out of "
    "hand. That's the real cost of ignoring philosophy: not ignorance, but a closed mind."
)


class TestPrompts:
    def test_registers_load_from_the_prompt_file(self):
        from generate_flat_training import load_llm_style_registers
        registers = load_llm_style_registers()
        assert {"explainer", "explainer_polished", "memo", "conversational", "punchy"} <= set(registers)
        for name, reg in registers.items():
            assert "{text}" in reg.steps[0] and "{words}" in reg.steps[0], name
            assert all("{text}" in step for step in reg.steps), name
            # Generic LLM styles; nothing author-specific in code or prompt.
            assert "Russell" not in "".join(reg.steps)

    def test_polished_explainer_is_the_explainer_then_a_polish(self):
        from generate_flat_training import load_llm_style_registers
        registers = load_llm_style_registers()
        polished = registers["explainer_polished"]
        assert polished.steps[0] == registers["explainer"].steps[0]
        assert len(polished.steps) == 2

    def test_device_registers_forbid_inventing_content(self):
        # Asked for a list of three, DeepSeek invented items to fill it.
        from generate_flat_training import load_llm_style_registers
        registers = load_llm_style_registers()
        for name in ("memo", "conversational", "punchy"):
            assert "never invent" in registers[name].steps[0].lower(), name

    def test_punchy_is_weighted_down(self):
        from generate_flat_training import load_llm_style_registers
        registers = load_llm_style_registers()
        assert registers["punchy"].weight < registers["explainer"].weight

    def test_steps_run_in_order(self, monkeypatch):
        import generate_flat_training as gft
        from generate_flat_training import LLMRegister
        prompts = []

        def fake(prompt, **kw):
            prompts.append(prompt)
            return REWRITE if len(prompts) == 2 else "DRAFT TEXT"

        monkeypatch.setattr(gft, "call_deepseek", fake)
        monkeypatch.setattr(gft, "load_llm_style_registers",
                            lambda: {"two": LLMRegister("two", ["A {words} {text}", "B {text}"], 1.0)})
        assert gft.llm_style_rewrite(RUSSELL, "two") == REWRITE
        assert prompts[0].startswith("A ") and RUSSELL in prompts[0]
        assert prompts[1] == "B DRAFT TEXT"


class TestRewriteCheck:
    def test_accepts_a_restructured_rewrite(self):
        from generate_flat_training import llm_rewrite_problem
        assert llm_rewrite_problem(RUSSELL, REWRITE) is None

    def test_rejects_an_echo(self):
        from generate_flat_training import llm_rewrite_problem
        assert "echo" in llm_rewrite_problem(RUSSELL, RUSSELL.replace("man", "person"))

    @pytest.mark.parametrize("bad", [
        "Here is the rewritten passage: " + REWRITE,
        "Here's a rewrite in that style: " + REWRITE,
        "Sure! " + REWRITE,
        "**Rewritten:** " + REWRITE,
    ])
    def test_rejects_meta_text(self, bad):
        from generate_flat_training import llm_rewrite_problem
        assert "meta" in llm_rewrite_problem(RUSSELL, bad)

    def test_a_register_opener_is_not_meta(self):
        from generate_flat_training import llm_rewrite_problem
        assert llm_rewrite_problem(RUSSELL, "Here's the thing: " + REWRITE) is None

    @pytest.mark.parametrize("text", ["Too short.", REWRITE * 3])
    def test_rejects_length_drift(self, text):
        from generate_flat_training import llm_rewrite_problem
        assert "length" in llm_rewrite_problem(RUSSELL, text)


class TestRTTJobs:
    def test_weighted_registers_are_distinct_per_chunk(self):
        import random
        from generate_flat_training import LLMRegister, rtt_jobs
        regs = {"a": LLMRegister("a", ["{text}{words}"], 1.0), "b": LLMRegister("b", ["{text}{words}"], 1.0),
                "c": LLMRegister("c", ["{text}{words}"], 0.1)}
        random.seed(0)
        seen = []
        for _ in range(300):
            jobs = rtt_jobs([(0, RUSSELL, "original", (1,))], lambda t, r: r, per_original=2, registers=regs)
            picked = [j.register for j in jobs if j.vtype == "llm_style"]
            assert len(set(picked)) == 2
            seen += picked
        assert seen.count("c") < seen.count("a") / 3

    def test_llm_style_only_skips_standard_and_other_types(self):
        from generate_flat_training import rtt_jobs
        batch = [(0, RUSSELL, "original", (3,)), (1, "Snowflake.", "snowflake", (5,))]
        jobs = rtt_jobs(batch, lambda t, r: "rewrite", per_original=2, registers=["x", "y"], include_standard=False)
        assert [(j.idx, j.vtype) for j in jobs] == [(0, "llm_style"), (0, "llm_style")]

    def test_originals_get_a_standard_row_and_llm_style_rows(self):
        from generate_flat_training import rtt_jobs
        batch = [(0, RUSSELL, "original", (3, 4)), (1, "Snowflake text here.", "snowflake", (5,))]
        calls = []

        def rewrite(text, register):
            calls.append(register)
            return f"{register} rewrite"

        jobs = rtt_jobs(batch, rewrite, per_original=2, registers=["punchy", "explainer", "memo"])
        kinds = [(j.idx, j.vtype) for j in jobs]
        assert kinds == [(0, "standard"), (0, "llm_style"), (0, "llm_style"), (1, "snowflake")]
        # The target is always the author's text; RTT runs on the rewrite.
        llm = [j for j in jobs if j.vtype == "llm_style"]
        assert all(j.styled == RUSSELL for j in llm)
        assert {j.rtt_source for j in llm} == {f"{r} rewrite" for r in calls}
        assert len(set(calls)) == 2  # two different registers
        assert all(j.register in calls for j in llm)

    def test_failed_rewrites_are_skipped(self):
        from generate_flat_training import rtt_jobs
        jobs = rtt_jobs([(0, RUSSELL, "original", (1,))], lambda t, r: None, per_original=2,
                        registers=["a", "b"])
        assert [j.vtype for j in jobs] == ["standard"]

    def test_llm_style_inputs_get_standard_noise(self, monkeypatch):
        import generate_flat_training as gft
        seen = []
        monkeypatch.setattr(gft, "perturb_text", lambda t, **k: seen.append(k) or t)
        gft.format_training_example("neutral words", "styled", "A", 2, variation_type="llm_style")
        assert seen == [{}]


class TestChunksMatchInferenceParagraphs:
    """Restyled paragraphs run ~60-250 words; 300-440 word chunks never
    showed the model a short paragraph's structure. Chunks still overlap:
    style lives in the transitions (docs/training_findings.md)."""

    def test_default_size(self):
        from generate_flat_training import OverlapConfig
        cfg = OverlapConfig()
        assert (cfg.min_words, cfg.max_words, cfg.overlap_sentences) == (100, 300, 2)

    def test_chunks_overlap_and_stay_in_range(self):
        from generate_flat_training import OverlapConfig, create_overlapping_chunks
        sentences = [f"Sentence number {i} says something about the matter at hand today." for i in range(200)]
        paragraphs = [(" ".join(sentences[i:i + 10]), "original", (i // 10,)) for i in range(0, 200, 10)]
        chunks = create_overlapping_chunks(paragraphs, OverlapConfig())
        words = [len(c[0].split()) for c in chunks]
        assert min(words) >= 100 and max(words) <= 330
        for (a, _, _), (b, _, _) in zip(chunks, chunks[1:]):
            last_two = a.split(". ")[-2:]
            assert all(s.rstrip(".") in b for s in last_two)

    def test_lengths_spread_across_the_range(self):
        # The window used to fill every chunk to max_words.
        import random
        from generate_flat_training import OverlapConfig, create_overlapping_chunks
        random.seed(0)
        sentences = [f"Sentence number {i} says something about the matter at hand today." for i in range(2000)]
        paragraphs = [(" ".join(sentences[i:i + 10]), "original", (i // 10,)) for i in range(0, 2000, 10)]
        words = [len(c[0].split()) for c in create_overlapping_chunks(paragraphs, OverlapConfig())]
        assert sum(w < 200 for w in words) / len(words) > 0.3
        assert sum(w > 250 for w in words) / len(words) > 0.15


class TestLLMStyleOnlyRun:
    """Regenerating only llm_style rows keeps every other row."""

    def test_replaces_llm_style_rows_and_keeps_the_rest(self, tmp_path, monkeypatch):
        import json
        from unittest.mock import MagicMock
        import generate_flat_training as gft
        out = tmp_path / "train.jsonl"
        old = [{"input": "a", "output": "A", "source_idx": 0, "source_paragraphs": [0], "variation_type": "standard"},
               {"input": "b", "output": "A", "source_idx": 0, "source_paragraphs": [0], "variation_type": "llm_style",
                "register": "punchy"},
               {"input": "c", "output": "C", "source_idx": 1, "source_paragraphs": [1], "variation_type": "snowflake"}]
        out.write_text("".join(json.dumps(r) + "\n" for r in old))
        neutralizer = MagicMock(spec=["neutralize"])
        neutralizer.neutralize.side_effect = lambda text, **kw: "plain words only here " + text
        monkeypatch.setattr(gft, "get_rtt_neutralizer", lambda: (neutralizer, MagicMock()))
        monkeypatch.setattr(gft, "check_lexical_bleed", lambda *a, **k: (True, 0.0))
        monkeypatch.setattr(gft, "llm_style_rewrite", lambda text, register: f"{register} rewrite")
        chunks = [("Styled chunk zero with enough words.", "original", (0,)),
                  ("Snowflake chunk one.", "snowflake", (1,))]
        gft.generate_training_data(chunks, "X", out, output_format="llama_factory", llm_style_only=True)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        kinds = [(r["source_idx"], r["variation_type"]) for r in rows]
        assert kinds[:2] == [(0, "standard"), (1, "snowflake")]
        assert kinds[2:] == [(0, "llm_style"), (0, "llm_style")]
        assert all(r.get("register") != "punchy" or r["input"] != "b" for r in rows)

    def test_keeps_other_rows_when_every_chunk_is_original(self, tmp_path, monkeypatch):
        import json
        from unittest.mock import MagicMock
        import generate_flat_training as gft
        out = tmp_path / "train.jsonl"
        out.write_text(json.dumps({"input": "a", "output": "A", "source_idx": 0, "source_paragraphs": [0],
                                   "variation_type": "standard"}) + "\n")
        neutralizer = MagicMock(spec=["neutralize"])
        neutralizer.neutralize.side_effect = lambda text, **kw: "plain words only here " + text
        monkeypatch.setattr(gft, "get_rtt_neutralizer", lambda: (neutralizer, MagicMock()))
        monkeypatch.setattr(gft, "check_lexical_bleed", lambda *a, **k: (True, 0.0))
        monkeypatch.setattr(gft, "llm_style_rewrite", lambda text, register: f"{register} rewrite")
        gft.generate_training_data([("Styled chunk zero with enough words.", "original", (0,))], "X", out,
                                   output_format="llama_factory", llm_style_only=True)
        kinds = [json.loads(line)["variation_type"] for line in out.read_text().splitlines()]
        assert kinds == ["standard", "llm_style", "llm_style"]

    def test_add_mode_keeps_existing_llm_style_rows(self, tmp_path, monkeypatch):
        import json
        from unittest.mock import MagicMock
        import generate_flat_training as gft
        out = tmp_path / "train.jsonl"
        old = {"input": "b", "output": "A", "source_idx": 0, "source_paragraphs": [0],
               "variation_type": "llm_style", "register": "memo"}
        out.write_text(json.dumps(old) + "\n")
        neutralizer = MagicMock(spec=["neutralize"])
        neutralizer.neutralize.side_effect = lambda text, **kw: "plain words only here " + text
        monkeypatch.setattr(gft, "get_rtt_neutralizer", lambda: (neutralizer, MagicMock()))
        monkeypatch.setattr(gft, "check_lexical_bleed", lambda *a, **k: (True, 0.0))
        monkeypatch.setattr(gft, "llm_style_rewrite", lambda text, register: f"{register} rewrite")
        gft.generate_training_data([("Styled chunk zero with enough words.", "original", (0,))], "X", out,
                                   output_format="llama_factory", llm_style_only=True, llm_style_add=True,
                                   llm_style_registers=["explainer", "explainer_polished"])
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert rows[0] == old
        assert sorted(r["register"] for r in rows[1:]) == ["explainer", "explainer_polished"]
