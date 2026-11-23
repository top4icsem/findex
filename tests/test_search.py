from pathlib import Path

import pytest

from findex.index import Posting, build_index
from findex.search import search


@pytest.fixture
def small_index(tmp_path: Path):
    """Create a small corpus with predictable search results."""
    documents = {
        "a.txt": "python search python",
        "b.txt": "python memory",
        "c.txt": "search engine",
        "d.txt": "memory engine",
    }

    for filename, text in documents.items():
        (tmp_path / filename).write_text(text, encoding="utf-8")

    return build_index(tmp_path)


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python AND search", [0]),
        ("python search", [0]),
        ("python OR search", [0, 1, 2]),
        ("NOT memory", [0, 2]),
        ("python AND NOT memory", [0]),
        ("(python OR search) AND NOT memory", [0, 2]),
        ("NOT (python OR search)", [3]),
        ("python OR search AND memory", [0, 1]),
        ("unknown", []),
        ("python AND unknown", []),
        ("python AND (search OR memory)", [0, 1]),
    ],
)
def test_boolean_search(small_index, engine, query, expected):
    result = search(small_index, query, engine=engine)
    assert result == expected


def test_posting_frequencies(small_index):
    assert small_index.postings["python"] == [
        Posting(doc_id=0, frequency=2),
        Posting(doc_id=1, frequency=1),
    ]


@pytest.mark.parametrize(
    "query",
    [
        "python AND",
        "(python OR search",
        "AND python",
    ],
)
def test_invalid_queries(small_index, query):
    with pytest.raises(ValueError):
        search(small_index, query)


def test_invalid_engine(small_index):
    with pytest.raises(ValueError):
        search(small_index, "python", engine="invalid")
