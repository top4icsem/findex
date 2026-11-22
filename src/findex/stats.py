
import argparse
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import perf_counter

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    top_terms: list[tuple[str, int]]
    elapsed_seconds: float
    peak_memory_mb: float


def collect_stats(root: Path, limit: int | None = None) -> CorpusStats:
    """Collect corpus statistics in a single pass."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    counts: Counter[str] = Counter()
    documents = 0
    tokens = 0

    tracemalloc.start()
    start = perf_counter()

    try:
        document_iterator = iter_documents(root)

        if limit is not None:
            document_iterator = islice(document_iterator, limit)

        for document in document_iterator:
            documents += 1

            for token in tokenize(document.text):
                counts[token] += 1
                tokens += 1

        elapsed = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return CorpusStats(
        documents=documents,
        tokens=tokens,
        vocabulary=len(counts),
        top_terms=counts.most_common(50),
        elapsed_seconds=elapsed,
        peak_memory_mb=peak / (1024 * 1024),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect statistics for a text corpus."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process.",
    )
    args = parser.parse_args()

    stats = collect_stats(Path("data"), limit=args.limit)

    print(f"Documents: {stats.documents}")
    print(f"Tokens: {stats.tokens}")
    print(f"Vocabulary: {stats.vocabulary}")
    print(f"Time: {stats.elapsed_seconds:.6f} seconds")
    print(f"Peak memory: {stats.peak_memory_mb:.4f} MB")

    print("\nTop 50 terms:")
    for term, count in stats.top_terms:
        print(f"{term}: {count}")

