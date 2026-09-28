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
        assert len(registers) >= 4
        for name, prompt in registers.items():
            assert "{text}" in prompt and "{words}" in prompt, name
            # Generic LLM styles; nothing author-specific in code or prompt.
            assert "Russell" not in prompt


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
