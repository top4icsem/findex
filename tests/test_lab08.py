"""Lab 08 regression tests: numeric parity, stable ties, semantic RRF and API."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from findex.index import DocMeta, Index, Posting
from findex.numpy_rank import NumpyIndex, NumpyRanker
from findex.ranking import RankedSearcher
from findex.scoring import BM25, Tfidf
from findex.semantic import EmbeddingStore, make_embeddings, reciprocal_rank_fusion
from findex.web.app import create_app, get_service
from findex.web.service import SearchService
from findex.web.settings import Settings


@pytest.fixture
def corpus_index() -> Index:
    return Index(
        postings={
            "python": [Posting(0, 3), Posting(3, 1), Posting(7, 2)],
            "search": [Posting(0, 1), Posting(3, 2)],
            "engine": [Posting(3, 1), Posting(7, 1)],
            "rare": [Posting(7, 1)],
        },
        documents={
            0: DocMeta(0, "a.txt", 8, "Python Python search"),
            3: DocMeta(3, "b.txt", 6, "Python search engine"),
            7: DocMeta(7, "c.txt", 12, "Python engine rare"),
            8: DocMeta(8, "d.txt", 4, "Unrelated"),
        },
    )


@pytest.mark.parametrize("scorer", [BM25(), Tfidf()])
@pytest.mark.parametrize(
    "query",
    [
        "python",
        "rare",
        "python AND search",
        "python OR engine",
        "NOT rare",
        "python AND (search OR engine)",
    ],
)
@pytest.mark.parametrize("k", [0, 1, 2, 3, 10])
def test_numpy_legacy_parity(
    corpus_index: Index, scorer: BM25 | Tfidf, query: str, k: int
) -> None:
    old = RankedSearcher(corpus_index, scorer).search(query, k)
    new = NumpyRanker(corpus_index, scorer).search(query, k)
    assert [item.doc_id for item in new] == [item.doc_id for item in old]
    assert [item.score for item in new] == pytest.approx(
        [item.score for item in old], abs=1e-9
    )


def test_columnar_int32_and_tie_break(corpus_index: Index) -> None:
    arrays = NumpyIndex(corpus_index)
    assert arrays.postings["python"].doc_ids.dtype == np.int32
    assert arrays.postings["python"].frequencies.dtype == np.int32
    assert arrays.doc_lengths.dtype == np.int32
    # For no positive query terms all scores tie; order must be doc_id ascending.
    assert [
        doc_id for doc_id, _score in NumpyRanker(corpus_index).rank_ids("NOT rare", 2)
    ] == [0, 3]


class FakeEncoder:
    def encode(self, sentences: str | list[str], **kwargs: object) -> np.ndarray:
        items = [sentences] if isinstance(sentences, str) else sentences
        vectors = []
        for item in items:
            low = item.lower()
            vectors.append(
                [1.0 if "python" in low else 0.0, 1.0 if "engine" in low else 0.0, 0.1]
            )
        return np.asarray(vectors, dtype=np.float32)


def test_embeddings_and_cosine(tmp_path: Path, corpus_index: Index) -> None:
    output = tmp_path / "vectors.npy"
    count, dim = make_embeddings(
        corpus_index, output, encoder=FakeEncoder(), chunk_size=16, overlap=0
    )
    assert (count, dim) == (4, 3)
    store = EmbeddingStore.load(output, corpus_index)
    store.encoder = FakeEncoder()
    ranked = store.rank_ids("python", 3)
    assert ranked[0][0] == 0
    assert np.linalg.norm(store.vectors[0]) == pytest.approx(1.0)
    assert json.loads(output.with_suffix(".json").read_text())["num_docs"] == 4


def test_rrf_prefers_both_ranks() -> None:
    results = reciprocal_rank_fusion([(0, 3.2), (1, 2.0)], [(1, 0.9), (2, 0.6)])
    assert results[0][0] == 1
    assert all(score > 0 for _id, score in results)


def test_semantic_missing_returns_422(corpus_index: Index) -> None:
    app = create_app(Settings(index_path=Path("/missing/lab08.json")))
    app.dependency_overrides[get_service] = lambda: SearchService(corpus_index)
    with TestClient(app) as client:
        assert (
            client.get(
                "/search", params={"q": "python", "mode": "semantic"}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/search", params={"q": "python", "mode": "notamode"}
            ).status_code
            == 422
        )
        assert (
            client.get("/search", params={"q": "python", "mode": "lexical"}).status_code
            == 200
        )
        assert 'name="mode"' in client.get("/").text
