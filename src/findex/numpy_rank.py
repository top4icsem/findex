"""Lab 08: vectorized BM25/TF-IDF ranking without changing Lab 01-07 storage.

Index.postings remains the canonical, backwards-compatible persistent format.
NumpyIndex builds an int32 columnar view once per loaded index, so scoring can
run over contiguous native arrays instead of per-Posting Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from findex.index import Index
from findex.query_ast import parse_query, positive_terms
from findex.ranking import SearchResult, make_snippet
from findex.scoring import BM25, Scorer

IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PostingColumns:
    doc_ids: IntArray
    frequencies: IntArray


class NumpyIndex:
    """Dense sorted document IDs, document lengths, and columnar term postings."""

    def __init__(self, index: Index):
        self.index = index
        ids = sorted(index.documents)
        if ids and (ids[0] < 0 or ids[-1] >= np.iinfo(np.int32).max):
            raise ValueError("doc IDs must fit a nonnegative int32")
        if any(index.documents[i].length > np.iinfo(np.int32).max for i in ids):
            raise ValueError("document length exceeds int32")
        self.doc_ids: IntArray = np.asarray(ids, dtype=np.int32)
        self.doc_lengths: IntArray = np.asarray(
            [index.documents[i].length for i in ids], dtype=np.int32
        )
        self.postings: dict[str, PostingColumns] = {}
        for term, values in index.postings.items():
            ordered = sorted(values, key=lambda p: p.doc_id)
            if any(
                p.frequency < 0 or p.frequency > np.iinfo(np.int32).max for p in ordered
            ):
                raise ValueError(f"Frequency for {term} cannot fit int32")
            self.postings[term] = PostingColumns(
                doc_ids=np.asarray([p.doc_id for p in ordered], dtype=np.int32),
                frequencies=np.asarray([p.frequency for p in ordered], dtype=np.int32),
            )


def top_indices(doc_ids: IntArray, scores: FloatArray, k: int) -> IntArray:
    """argpartition first, then deterministic (-score, doc_id) tie-breaking.

    Include *all* boundary ties before sorting, otherwise argpartition could
    silently select different documents than Lab 03's heapq.nlargest.
    """
    count = len(doc_ids)
    if k <= 0 or count == 0:
        return np.empty(0, dtype=np.int32)
    if k >= count:
        candidates = np.arange(count, dtype=np.int32)
    else:
        split = np.argpartition(-scores, k - 1)[:k]
        threshold = np.min(scores[split])
        candidates = np.flatnonzero(scores >= threshold).astype(np.int32)
    order = np.lexsort((doc_ids[candidates], -scores[candidates]))
    return candidates[order[:k]]


class NumpyRanker:
    """Vectorized scoring with the exact Boolean filter and tie policy of Lab 03."""

    def __init__(self, index: Index, scorer: Scorer | None = None):
        self.index = index
        self.scorer: Scorer = scorer if scorer is not None else BM25()
        self.arrays = NumpyIndex(index)

        @lru_cache(maxsize=256)
        def matching_ids(query: str) -> tuple[int, ...]:
            return tuple(sorted(parse_query(query).evaluate(index)))

        self._matching_ids = matching_ids

    def rank_ids(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Scores only, without file I/O from snippets; useful for benchmarks."""
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        candidates = self._matching_ids(query)
        if top_k == 0 or not candidates:
            return []
        doc_ids: IntArray = np.asarray(candidates, dtype=np.int32)
        positions = np.searchsorted(self.arrays.doc_ids, doc_ids)
        lengths = self.arrays.doc_lengths[positions].astype(np.float64)
        scores: FloatArray = np.zeros(len(doc_ids), dtype=np.float64)
        terms = positive_terms(parse_query(query))
        n = self.index.num_docs
        avg = max(self.index.avg_doc_length, 1.0)
        for term in terms:
            posting = self.arrays.postings.get(term)
            if posting is None or len(posting.doc_ids) == 0:
                continue
            matches = np.searchsorted(doc_ids, posting.doc_ids)
            valid = matches < len(doc_ids)
            matches = matches[valid]
            freqs = posting.frequencies[valid]
            valid = doc_ids[matches] == posting.doc_ids[valid]
            if not np.any(valid):
                continue
            positions_in_candidate = matches[valid]
            tf = freqs[valid].astype(np.float64)
            df = self.index.df(term)
            if isinstance(self.scorer, BM25):
                idf = np.log1p((n - df + 0.5) / (df + 0.5))
                norm = self.scorer.k1 * (
                    1.0
                    - self.scorer.b
                    + self.scorer.b * lengths[positions_in_candidate] / avg
                )
                value = idf * tf * (self.scorer.k1 + 1.0) / (tf + norm)
            else:
                idf = np.log((n + 1) / (df + 1)) + 1.0
                value = (
                    (1.0 + np.log(tf))
                    * idf
                    / np.sqrt(np.maximum(1.0, lengths[positions_in_candidate]))
                )
            np.add.at(scores, positions_in_candidate, value)
        best = top_indices(doc_ids, scores, top_k)
        return [(int(doc_ids[i]), float(scores[i])) for i in best]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        terms = positive_terms(parse_query(query))
        return [
            SearchResult(
                doc_id=doc_id,
                score=score,
                path=self.index.documents[doc_id].path,
                snippet=make_snippet(self.index, doc_id, terms),
            )
            for doc_id, score in self.rank_ids(query, top_k)
        ]
