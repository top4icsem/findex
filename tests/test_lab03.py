import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from findex.index import Index, Posting, build_index
from findex.open_index import open_index
from findex.query_ast import And, Not, Or, Phrase, Term, parse_query, positive_terms
from findex.ranking import RankedSearcher, SearchResult
from findex.scoring import BM25, Tfidf
from findex.storage import load_index, save_index
from findex.timing import timed


@pytest.fixture
def tiny_index(tmp_path: Path) -> Index:
    (tmp_path / "a.txt").write_text("python search event loop", encoding="utf-8")
    (tmp_path / "b.txt").write_text("python python event loop", encoding="utf-8")
    (tmp_path / "c.txt").write_text("java event driven loop", encoding="utf-8")
    (tmp_path / "d.txt").write_text("python async await event loop", encoding="utf-8")
    return build_index(tmp_path, with_positions=True)


def test_data_model(tiny_index):
    assert len(tiny_index) == len(tiny_index.postings)
    assert tiny_index.num_docs == 4
    assert tiny_index.df("python") == 3
    assert "python" in tiny_index
    assert tiny_index["python"] == tiny_index.postings["python"]
    assert "event" in list(tiny_index)
    assert "num_docs=4" in repr(tiny_index)
    assert tiny_index.avg_doc_length > 0
    with pytest.raises(FrozenInstanceError):
        Posting(1, 1).frequency = 2


def test_ast_structure():
    actual = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert actual == expected
    assert (Term("python") & ~Term("java")) == And(Term("python"), Not(Term("java")))
    assert positive_terms(actual) == {"python", "async", "await", "event", "loop"}


def test_phrase(tiny_index):
    assert parse_query('"event loop"').evaluate(tiny_index) == {0, 1, 3}
    assert parse_query('python "event loop"').evaluate(tiny_index) == {0, 1, 3}
    assert parse_query('"event driven"').evaluate(tiny_index) == {2}
    assert parse_query('python AND (async OR await) NOT java "event loop"').evaluate(
        tiny_index
    ) == {3}


def test_phrase_requires_positions(tmp_path: Path):
    (tmp_path / "a.txt").write_text("event loop", encoding="utf-8")
    plain = build_index(tmp_path)
    with pytest.raises(ValueError, match="positions"):
        parse_query('"event loop"').evaluate(plain)


@pytest.mark.parametrize("format", ["pickle", "json"])
def test_storage_phrase_roundtrip(tiny_index, tmp_path: Path, format: str):
    path = tmp_path / ("index.bin" if format == "pickle" else "index.json")
    save_index(tiny_index, path, format)
    restored = load_index(path, format)
    assert restored == tiny_index
    assert parse_query('"event loop"').evaluate(restored) == {0, 1, 3}


def test_ranked_results_and_cache(tiny_index, caplog):
    searcher = RankedSearcher(tiny_index, BM25())
    with caplog.at_level(logging.INFO):
        result1 = searcher.search("python", top_k=2)
        result2 = searcher.search("python", top_k=2)
    assert result1 == result2
    assert len(result1) == 2
    assert result1 == sorted(result1, reverse=True)
    assert searcher._matching_ids.cache_info().hits == 1
    assert "CACHE HIT" in caplog.text
    assert "**python**" in result1[0].snippet


def test_tfidf_is_interchangeable(tiny_index):
    bm = RankedSearcher(tiny_index, BM25())
    tfidf = RankedSearcher(tiny_index, Tfidf())
    assert {r.doc_id for r in bm.search("python", top_k=10)} == {
        r.doc_id for r in tfidf.search("python", top_k=10)
    }


def test_rank_order():
    high = SearchResult(3, 9.0, "high", "yes")
    low = SearchResult(1, 1.0, "low", "no")
    assert high > low


def test_context_manager_closes_on_exception(tiny_index, tmp_path):
    path = tmp_path / "index.json"
    save_index(tiny_index, path, "json")
    with pytest.raises(RuntimeError), open_index(path) as opened:
        assert opened.num_docs == 4
        raise RuntimeError("failure")
    assert opened.num_docs == 0
    assert len(opened) == 0


def test_bad_queries():
    for query in ('"unterminated', "python AND", "(python OR java", ""):
        with pytest.raises(ValueError):
            parse_query(query)


def test_timed_preserves_metadata():
    @timed
    def square(value):
        """Square a number."""
        return value * value

    assert square.__name__ == "square"
    assert square.__doc__ == "Square a number."
    assert square(3) == 9
