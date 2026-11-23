"""Shared, small test corpus independent of the generated 1000-file dataset."""

from pathlib import Path

import pytest

from findex.index import Index, build_index


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("python search event loop", encoding="utf-8")
    (corpus / "b.txt").write_text("python python memory", encoding="utf-8")
    (corpus / "c.txt").write_text("java machine learning", encoding="utf-8")
    return corpus


@pytest.fixture
def sample_index(sample_corpus: Path) -> Index:
    return build_index(sample_corpus, with_positions=True)
