import argparse
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    length: int
    preview: str


@dataclass
class InvertedIndex:
    postings: dict[str, list[Posting]]
    documents: dict[int, DocMeta]


def build_index(root: Path) -> InvertedIndex:
    """Build sorted posting lists in one pass through the corpus."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}

    for doc_id, document in enumerate(iter_documents(root)):
        frequencies = Counter(tokenize(document.text))

        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            path=document.path.relative_to(root).as_posix(),
            length=sum(frequencies.values()),
            preview=document.text[:200],
        )

        for term, frequency in frequencies.items():
            postings[term].append(Posting(doc_id=doc_id, frequency=frequency))

    return InvertedIndex(
        postings=dict(postings),
        documents=documents,
    )


def main() -> None:
    from findex.index import build_index as canonical_build_index
    from findex.storage import save_index

    parser = argparse.ArgumentParser(description="Build and save the findex index.")
    parser.add_argument(
        "corpus",
        type=Path,
        help="Directory containing .txt files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output index file",
    )
    parser.add_argument(
        "--format",
        choices=("pickle", "json"),
        default=None,
    )
    args = parser.parse_args()

    file_format = args.format or ("json" if args.out.suffix == ".json" else "pickle")

    tracemalloc.start()
    start = perf_counter()

    try:
        # Use canonical classes for cross-process pickle compatibility.
        index = canonical_build_index(args.corpus)
        built = perf_counter()

        save_index(index, args.out, file_format)
        finished = perf_counter()

        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Documents: {len(index.documents)}")
    print(f"Vocabulary: {len(index.postings)}")
    print(f"Format: {file_format}")
    print(f"Saved: {args.out}")
    print(f"Size: {args.out.stat().st_size} bytes")
    print(f"Indexing time: {built - start:.4f} s")
    print(f"Save time: {finished - built:.4f} s")
    print(f"Peak memory: {peak / 1024**2:.4f} MiB")


if __name__ == "__main__":
    main()
