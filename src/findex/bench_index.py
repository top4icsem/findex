import argparse
import gc
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

from findex.corpus import iter_documents
from findex.index import Index, Posting, build_index
from findex.search import merge_and
from findex.storage import load_index, save_index
from findex.tokenize import tokenize


@dataclass(frozen=True)
class RegularPosting:
    doc_id: int
    frequency: int


type PostingVariants = (
    defaultdict[str, list[RegularPosting]]
    | defaultdict[str, list[Posting]]
    | defaultdict[str, tuple[array[int], array[int]]]
)

type MemoryRow = tuple[str, int, int, int, float, float, float]


def build_posting_variant(root: Path, kind: str) -> tuple[PostingVariants, int, int]:
    """Build precisely one variant of postings, without document metadata."""
    regular_postings: defaultdict[str, list[RegularPosting]] = defaultdict(list)
    slot_postings: defaultdict[str, list[Posting]] = defaultdict(list)
    array_postings: defaultdict[str, tuple[array[int], array[int]]] = defaultdict(
        lambda: (array("I"), array("I"))
    )
    if kind not in {"regular", "slots", "array"}:
        raise ValueError(f"Unknown variant: {kind}")

    documents = 0
    for doc_id, document in enumerate(iter_documents(root)):
        documents += 1
        frequencies: Counter[str] = Counter(tokenize(document.text))
        for term, frequency in frequencies.items():
            if kind == "regular":
                regular_postings[term].append(RegularPosting(doc_id, frequency))
            elif kind == "slots":
                slot_postings[term].append(Posting(doc_id, frequency))
            else:
                ids, counts = array_postings[term]
                ids.append(doc_id)
                counts.append(frequency)

    if kind == "regular":
        postings: PostingVariants = regular_postings
        count = sum(map(len, regular_postings.values()))
    elif kind == "slots":
        postings = slot_postings
        count = sum(map(len, slot_postings.values()))
    else:
        postings = array_postings
        count = sum(len(pair[0]) for pair in array_postings.values())

    return postings, documents, count


def benchmark_memory(root: Path) -> None:
    rows: list[MemoryRow] = []

    for kind in ("regular", "slots", "array"):
        gc.collect()
        tracemalloc.start()
        start = perf_counter()

        try:
            postings, documents, count = build_posting_variant(root, kind)
            elapsed = perf_counter() - start
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        rows.append(
            (
                kind,
                documents,
                len(postings),
                count,
                current / 1024**2,
                peak / 1024**2,
                elapsed,
            )
        )

        del postings
        gc.collect()

    assert len({(row[1], row[2], row[3]) for row in rows}) == 1

    print("\nMEMORY (postings only, MiB)")
    print("kind       documents  terms  pairs    retained_MiB  peak_MiB  time_s")

    for kind, documents, terms, count, current, peak, elapsed in rows:
        print(
            f"{kind:<10} {documents:>9} {terms:>6}"
            f" {count:>7} {current:>12.3f}"
            f" {peak:>9.3f} {elapsed:>7.3f}"
        )


def time_per_call(action: Callable[[], object], repeats: int = 1000) -> float:
    samples: list[float] = []

    for _ in range(5):
        start = perf_counter()

        for _ in range(repeats):
            action()

        samples.append((perf_counter() - start) * 1000 / repeats)

    return median(samples)


def benchmark_search(index: Index, repeats: int) -> None:
    totals = {
        term: sum(p.frequency for p in posting_list)
        for term, posting_list in index.postings.items()
    }

    frequent = sorted(
        totals,
        key=lambda term: (-totals[term], term),
    )[:2]

    rare = sorted(
        totals,
        key=lambda term: (totals[term], term),
    )[:2]

    print("\nSEARCH (AND intersection, median ms per call)")
    print("case       term_1     term_2     matches     merge_ms      set_ms")

    for label, terms in (
        ("frequent", frequent),
        ("rare", rare),
    ):
        left = [p.doc_id for p in index.postings[terms[0]]]
        right = [p.doc_id for p in index.postings[terms[1]]]

        # Bind the current iteration explicitly; avoids a late-bound closure.
        def merge_fn(left: list[int] = left, right: list[int] = right) -> list[int]:
            return merge_and(left, right)

        def set_fn(left: list[int] = left, right: list[int] = right) -> list[int]:
            return sorted(set(left) & set(right))

        matches = merge_fn()
        assert matches == set_fn()

        merge_ms = time_per_call(merge_fn, repeats)
        set_ms = time_per_call(set_fn, repeats)

        print(
            f"{label:<10} {terms[0]:<10}"
            f" {terms[1]:<10} {len(matches):>7}"
            f" {merge_ms:>12.6f} {set_ms:>11.6f}"
        )


def benchmark_storage(index: Index, root: Path, repeats: int = 3) -> None:
    print("\nSTORAGE (median ms, file size in MiB)")
    print("format       file_MiB     save_ms     load_ms   verified")

    for file_format, extension in (
        ("pickle", "bin"),
        ("json", "json"),
    ):
        path = root / f"_bench_index.{extension}"
        save_samples: list[float] = []
        load_samples: list[float] = []

        for _ in range(repeats):
            start = perf_counter()
            save_index(index, path, file_format)
            save_samples.append((perf_counter() - start) * 1000)

            start = perf_counter()
            loaded = load_index(path, file_format)
            load_samples.append((perf_counter() - start) * 1000)

            assert loaded == index
            del loaded

        size = path.stat().st_size / 1024**2

        print(
            f"{file_format:<10} {size:>10.4f}"
            f" {median(save_samples):>11.4f}"
            f" {median(load_samples):>11.4f}"
            "       True"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the inverted index.")
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be positive")

    benchmark_memory(args.corpus)

    index = build_index(args.corpus)

    benchmark_search(index, args.iterations)
    benchmark_storage(index, args.corpus)


if __name__ == "__main__":
    main()
