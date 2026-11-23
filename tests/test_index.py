from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.index import DocMeta, Posting, build_index


def test_index_contents(tmp_path: Path):
    (tmp_path / "b.txt").write_text("search engine", encoding="utf-8")
    (tmp_path / "a.txt").write_text("python python search", encoding="utf-8")

    idx = build_index(tmp_path)

    assert len(idx.documents) == 2
    assert idx.documents[0].path == "a.txt"
    assert idx.documents[0].length == 3

    assert idx.postings["python"] == [
        Posting(0, 2),
    ]
    assert idx.postings["search"] == [
        Posting(0, 1),
        Posting(1, 1),
    ]


def test_sorted_postings(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"{i}.txt").write_text("shared term", encoding="utf-8")

    idx = build_index(tmp_path)

    assert [posting.doc_id for posting in idx.postings["shared"]] == list(range(5))


def test_immutable_slots():
    assert hasattr(Posting, "__slots__")
    assert hasattr(DocMeta, "__slots__")

    with pytest.raises(FrozenInstanceError):
        Posting(1, 2).frequency = 3
