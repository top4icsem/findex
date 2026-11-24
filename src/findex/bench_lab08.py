"""Measured Lab 03 Python vs Lab 08 NumPy scoring, no snippet-file I/O.

Run: uv run python -m findex.bench_lab08 --index index.bin --repeat 30
Saves docs/lab08_scoring.csv and docs/lab08_scoring.md with real timings.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import statistics
import time
from pathlib import Path

from findex.index import Index
from findex.numpy_rank import NumpyRanker
from findex.open_index import open_index
from findex.query_ast import parse_query, positive_terms
from findex.scoring import BM25, Scorer, Tfidf


def legacy_score_ids(
    index: Index, query: str, scorer: Scorer, top_k: int
) -> list[tuple[int, float]]:
    """Copy numeric semantics of Lab 03 without the snippet disk I/O."""
    ids = sorted(parse_query(query).evaluate(index))
    terms = positive_terms(parse_query(query))
    by_term = {
        term: {posting.doc_id: posting.frequency for posting in index[term]}
        for term in terms
        if term in index
    }
    scored: list[tuple[float, int, int]] = []
    for doc_id in ids:
        meta = index.documents[doc_id]
        score = sum(
            scorer.score(
                tf=by_doc.get(doc_id, 0),
                df=index.df(term),
                doc_length=meta.length,
                avg_doc_length=index.avg_doc_length,
                num_docs=index.num_docs,
            )
            for term, by_doc in by_term.items()
        )
        scored.append((score, -doc_id, doc_id))
    return [(doc_id, score) for score, _neg_id, doc_id in heapq.nlargest(top_k, scored)]


def milliseconds(fn: object, repeat: int) -> float:
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()  # type: ignore[operator]
        times.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(times)


def choose_queries(index: Index) -> list[tuple[str, str]]:
    available = [
        (t, len(postings)) for t, postings in index.postings.items() if postings
    ]
    if not available:
        raise ValueError("Empty index")
    frequent = max(available, key=lambda row: (row[1], row[0]))[0]
    rare = min(available, key=lambda row: (row[1], row[0]))[0]
    others = [
        term
        for term, _count in sorted(available, key=lambda r: -r[1])
        if term != frequent
    ]
    triple = " OR ".join([frequent] + others[:2])
    return [("frequent", frequent), ("rare", rare), ("three_terms", triple)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 03 vs NumPy scoring medians")
    parser.add_argument("--index", type=Path, default=Path("index.bin"))
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("docs/lab08_scoring.csv"))
    args = parser.parse_args()
    if args.repeat < 3:
        parser.error("Use at least three repetitions")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with open_index(args.index) as index:
        for score_name, scorer in (("BM25", BM25()), ("TF-IDF", Tfidf())):
            numpy_ranker = NumpyRanker(index, scorer)
            for label, query in choose_queries(index):
                baseline = legacy_score_ids(index, query, scorer, 5)
                vector = numpy_ranker.rank_ids(query, 5)
                if [doc for doc, _ in baseline] != [doc for doc, _ in vector]:
                    raise AssertionError(f"Ranking mismatch: {score_name} {query}")
                for (_doc_a, a), (_doc_b, b) in zip(baseline, vector):
                    if abs(a - b) > 1e-8:
                        raise AssertionError("Scorer numerical parity mismatch")
                # Cache warmup kept in both paths.
                legacy_score_ids(index, query, scorer, 5)
                numpy_ranker.rank_ids(query, 5)
                old = milliseconds(
                    lambda index=index, query=query, scorer=scorer: legacy_score_ids(
                        index, query, scorer, 5
                    ),
                    args.repeat,
                )
                new = milliseconds(
                    lambda numpy_ranker=numpy_ranker, query=query: (
                        numpy_ranker.rank_ids(query, 5)
                    ),
                    args.repeat,
                )
                rows.append(
                    {
                        "scorer": score_name,
                        "query_type": label,
                        "query": query,
                        "python_ms": f"{old:.6f}",
                        "numpy_ms": f"{new:.6f}",
                        "speedup": f"{old / new:.3f}" if new > 0 else "nan",
                    }
                )
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = "| Скорер | Запит | Python, мс | NumPy, мс | Прискорення |\n|---|---|---:|---:|---:|\n"
    for row in rows:
        md += (
            f"| {row['scorer']} | {row['query_type']} | {float(row['python_ms']):.4f} "
            f"| {float(row['numpy_ms']):.4f} | {row['speedup']}× |\n"
        )
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print("Saved", args.out, "and", args.out.with_suffix(".md"))


if __name__ == "__main__":
    main()
