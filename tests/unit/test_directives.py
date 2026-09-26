"""Style directives: per-author, checkable, and drawn from a real paragraph.

Training takes a row's directives from its target; inference takes them from
the grafted exemplar. Both are real paragraphs by the author, so the model
sees the same kind of directive sets in both and every one is true of the
text it was trained to produce.
"""

import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest

LONG = ("It is generally recognized that he has revolutionized our conception of the physical world, "
        "but his new conceptions are wrapped up in mathematical technicalities which few can follow at all")
TARGET = (f"{LONG}; the public admires him without understanding him. Suppose, for example, that a man "
          "is told the earth moves (he has always seen it stand still). He will not believe it. "
          "But we must not blame him.")


def _directive_lines(text):
    return [l[len("[CONSTRAINT]: "):] for l in text.splitlines() if l.startswith("[CONSTRAINT]")]


class TestChecks:
    @pytest.mark.parametrize("check, good, bad", [
        ("opens_long", LONG + ". Then.", "Short start. " + LONG + "."),
        ("opens_short", "He was wrong. " + LONG + ".", LONG + "."),
        ("long_sentence", LONG + " and a few more words here and there to pass forty.", "He left."),
        ("short_after_long", LONG + ". He was wrong.", "He was wrong. " + LONG + "."),
        ("ends_short", LONG + ". He was wrong.", "He was wrong. " + LONG + "."),
        ("semicolon", "He left; she stayed.", "He left."),
        ("colon", "One thing matters: truth.", "He left."),
        ("parenthesis", "He (oddly) left.", "He left."),
        ("dash", "He left — oddly.", "He left."),
        ("question", "Why? Because.", "Because."),
        ("example", "Take, for example, a stone.", "A stone."),
        ("hypothetical", "Suppose a man is told this.", "A man is told this."),
        ("not_but", "It is not wisdom but habit.", "It is habit."),
        ("conjunction_start", "He left. But she stayed.", "He left. She stayed."),
        ("we", "We must not blame him.", "One must not blame him."),
        ("first_person", "I think he was wrong.", "He was wrong."),
        ("scare_quotes", 'What he calls "matter" is a fiction.', "Matter is a fiction."),
        ("concession", "No doubt he meant well.", "He meant well."),
    ])
    def test_check(self, check, good, bad):
        from src.persona.prompt_builder import DIRECTIVE_CHECKS
        assert DIRECTIVE_CHECKS[check](good)
        assert not DIRECTIVE_CHECKS[check](bad)


class TestPersonaFile:
    def test_russell_directives_load_with_known_checks(self):
        from src.persona.prompt_builder import _load_persona_file, DIRECTIVE_CHECKS
        directives = _load_persona_file("russell_worldview.txt")["directives"]
        assert len(directives) >= 10
        assert all(check in DIRECTIVE_CHECKS and text for check, text in directives)

    def test_unknown_check_is_an_error(self, tmp_path):
        from src.persona import prompt_builder as pb
        (tmp_path / "x.txt").write_text("[PERSONA_FRAMES_CONCEPTUAL]\nFrame\n\n[DIRECTIVES]\nnope: Do a thing.\n")
        with patch.object(pb, "_PROMPTS_DIR", tmp_path), pytest.raises(ValueError, match="nope"):
            pb._load_persona_file("x.txt")

    def test_files_without_directives_have_none(self):
        from src.persona.prompt_builder import _load_persona_file
        assert _load_persona_file("lovecraft_worldview.txt")["directives"] == []


class TestInstruction:
    def _build(self, **kw):
        from src.persona.prompt_builder import build_persona_instruction
        return build_persona_instruction("neutral input words", worldview="russell_worldview.txt", **kw)

    def test_training_directives_are_true_of_the_target(self):
        from src.persona.prompt_builder import _load_persona_file, DIRECTIVE_CHECKS
        checks = dict((t, c) for c, t in _load_persona_file("russell_worldview.txt")["directives"])
        random.seed(0)
        for _ in range(50):
            lines = [l for l in _directive_lines(self._build(satisfied_by=TARGET)) if l in checks]
            assert 2 <= len(lines) <= 4
            assert all(DIRECTIVE_CHECKS[checks[l]](TARGET) for l in lines)

    def test_directive_sets_vary(self):
        random.seed(0)
        sets = {tuple(_directive_lines(self._build(satisfied_by=TARGET))) for _ in range(30)}
        assert len(sets) > 5

    def test_inference_draws_from_the_grafted_exemplar(self):
        graft = SimpleNamespace(skeleton=None, sample_text=TARGET)
        random.seed(0)
        with_graft = self._build(grafting_guidance=graft)
        random.seed(0)
        assert with_graft == self._build(satisfied_by=TARGET, grafting_guidance=graft)

    def test_old_tiers_are_gone_for_authors_with_directives(self):
        from src.persona.prompt_builder import FREQUENT_CONSTRAINTS, ROTATING_CONSTRAINTS
        random.seed(0)
        for _ in range(30):
            lines = _directive_lines(self._build(satisfied_by=TARGET))
            assert not set(lines) & set(FREQUENT_CONSTRAINTS + ROTATING_CONSTRAINTS)

    def test_generic_constraints_still_apply_when_obeyed(self):
        lines = _directive_lines(self._build(satisfied_by=TARGET))
        assert any(l.startswith("Do not hedge") for l in lines)

    def test_authors_without_directives_keep_the_old_tiers(self):
        from src.persona.prompt_builder import build_persona_instruction
        random.seed(1)
        texts = [build_persona_instruction("input", worldview="lovecraft_worldview.txt") for _ in range(20)]
        assert any("topic sentence" in t for t in texts)
