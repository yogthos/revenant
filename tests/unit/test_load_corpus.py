"""scripts/load_corpus.py must not index without skeletons when it can't make them."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def test_missing_llm_config_fails_before_indexing(tmp_path):
    from load_corpus import load_corpus
    corpus = tmp_path / "c.txt"
    corpus.write_text(("This is a perfectly ordinary paragraph of prose with enough words in it to pass "
                       "every quality check that the loader applies to its input text. ") * 2)
    indexer = MagicMock()
    with patch("src.config.load_config", side_effect=FileNotFoundError("Configuration file not found")), \
         patch("src.rag.corpus_indexer.get_indexer", return_value=indexer):
        with pytest.raises(FileNotFoundError):
            load_corpus(str(corpus), "A", extract_skeletons=True)
    indexer.collection.add.assert_not_called()
    indexer.embedding_model.encode.assert_not_called()


def test_skeletons_run_in_parallel_and_keep_order(monkeypatch):
    import threading
    from load_corpus import extract_skeletons_batch
    from src.rag import skeleton_extractor
    from src.rag.skeleton_extractor import ArgumentSkeleton

    threads = set()

    def fake(chunk, provider):
        threads.add(threading.get_ident())
        if chunk == "bad":
            raise RuntimeError("api error")
        return ArgumentSkeleton(moves=[chunk], raw="")

    monkeypatch.setattr(skeleton_extractor, "extract_skeleton", fake)
    chunks = [f"c{i}" for i in range(40)] + ["bad"]
    out = extract_skeletons_batch(chunks, object(), workers=8)
    assert out[:40] == [ArgumentSkeleton(moves=[f"c{i}"], raw="").to_metadata() for i in range(40)]
    assert out[40] is None
    assert len(threads) > 1
