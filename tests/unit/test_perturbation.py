"""Tests for input perturbation (shared by training and inference)."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


ARTICLES = {"the", "a", "an"}


def _run_recording_drops(monkeypatch, fn, text, **kwargs):
    """Run ``fn`` and return (output, words it chose to drop)."""
    from src.utils import perturbation
    dropped = []
    original = perturbation._is_droppable

    def spy(word):
        result = original(word)
        if result:
            dropped.append(word)
        return result

    monkeypatch.setattr(perturbation, "_is_droppable", spy)
    return fn(text, **kwargs), dropped


class TestPerturbationKeepsMeaning:
    SAMPLE = (
        "It is always hard, especially when you are not young anymore. "
        "Great men never doubt; only little minds are certain. The old house was very small."
    )

    @pytest.mark.parametrize("seed", range(50))
    def test_only_articles_are_dropped(self, seed, monkeypatch):
        from src.utils.perturbation import perturb_text
        random.seed(seed)
        out, dropped = _run_recording_drops(monkeypatch, perturb_text, self.SAMPLE, perturbation_rate=0.5)
        assert len(out.split()) == len(self.SAMPLE.split()) - len(dropped)
        assert {w.lower() for w in dropped} <= ARTICLES

    def test_meaning_words_never_dropped(self):
        from src.utils.perturbation import perturb_text
        for seed in range(200):
            random.seed(seed)
            out = perturb_text(self.SAMPLE, perturbation_rate=0.3).lower()
            letters = [sorted(w.strip(".,;:!?")) for w in out.split()]
            for word in ("never", "only", "always", "not"):
                # Present, possibly with two letters swapped by a typo.
                assert sorted(word) in letters, f"seed {seed} dropped {word!r}: {out}"

    def test_is_droppable_only_for_bare_articles(self):
        from src.utils.perturbation import _is_droppable
        assert _is_droppable("the") and _is_droppable("An")
        for word in ("never", "only", "hard", "young", "great", "little", "very", "the."):
            assert not _is_droppable(word)

    def test_no_adjective_drop_option(self):
        import inspect
        from src.utils.perturbation import perturb_text
        assert "drop_adjectives" not in inspect.signature(perturb_text).parameters

    def test_synonym_swap_keeps_punctuation(self):
        from src.utils.perturbation import perturb_text
        random.seed(0)
        for _ in range(200):
            out = perturb_text("It was big.", perturbation_rate=1.0)
            assert out.endswith(".")

    def test_rate_zero_is_identity(self):
        from src.utils.perturbation import perturb_text
        assert perturb_text(self.SAMPLE, perturbation_rate=0.0) == self.SAMPLE


class TestHeavyPerturbation:
    @pytest.mark.parametrize("seed", range(30))
    def test_heavy_only_drops_articles(self, seed, monkeypatch):
        from src.utils.perturbation import heavy_perturb_text
        random.seed(seed)
        text = TestPerturbationKeepsMeaning.SAMPLE
        out, dropped = _run_recording_drops(monkeypatch, heavy_perturb_text, text, perturbation_rate=0.6)
        assert len(out.split()) == len(text.split()) - len(dropped)
        assert {w.lower() for w in dropped} <= ARTICLES


class TestTrainingUsesSharedPerturbation:
    def test_training_script_imports_shared_functions(self):
        import generate_flat_training as gft
        from src.utils import perturbation
        assert gft.perturb_text is perturbation.perturb_text
        assert gft.create_heavy_perturbation is perturbation.heavy_perturb_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
