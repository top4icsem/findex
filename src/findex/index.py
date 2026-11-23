"""The inverted index, Python data model methods, and indexing CLI."""

import argparse
import logging
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from time import perf_counter

from findex.corpus import iter_documents
from findex.timing import timed
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    length: int
    preview: str


@dataclass
class Index:
    postings: dict[str, list[Posting]]
    documents: dict[int, DocMeta]
    with_positions: bool = False
    corpus_root: str = ""

    def __len__(self) -> int:
        """Number of terms, like a Mapping."""
        return len(self.postings)

    def __iter__(self) -> Iterator[str]:
        """Iterate over terms."""
        return iter(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __repr__(self) -> str:
        return (
            f"Index(num_docs={self.num_docs}, terms={len(self)}, "
            f"with_positions={self.with_positions})"
        )

    @property
    def num_docs(self) -> int:
        return len(self.documents)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.documents:
            return 0.0
        return sum(meta.length for meta in self.documents.values()) / self.num_docs

    def df(self, term: str) -> int:
        """Document frequency of a term."""
        return len(self.postings.get(term, ()))

    def close(self) -> None:
        """Release large dictionaries when the context closes."""
        self.postings.clear()
        self.documents.clear()
        self.__dict__.pop("avg_doc_length", None)


# Backwards compatible with Lab 02 imports and tests.
InvertedIndex = Index


@timed
def build_index(
    root: Path,
    with_positions: bool = False,
    progress: Callable[[int], None] | None = None,
) -> Index:
    """Build one-pass postings; token positions are optional."""
    root = Path(root)
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}

    for doc_id, document in enumerate(iter_documents(root)):
        if with_positions:
            positions_by_term: defaultdict[str, list[int]] = defaultdict(list)
            for position, term in enumerate(tokenize(document.text)):
                positions_by_term[term].append(position)
            frequencies = {
                term: len(positions) for term, positions in positions_by_term.items()
            }
        else:
            positions_by_term = defaultdict(list)
            frequencies = Counter(tokenize(document.text))

        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            path=document.path.relative_to(root).as_posix(),
            length=sum(frequencies.values()),
            preview=document.text[:200],
        )

        for term, frequency in frequencies.items():
            postings[term].append(
                Posting(
                    doc_id=doc_id,
                    frequency=frequency,
                    positions=tuple(positions_by_term[term]) if with_positions else (),
                )
            )

        if progress is not None:
            progress(doc_id + 1)

    return Index(
        postings=dict(postings),
        documents=documents,
        with_positions=with_positions,
        corpus_root=str(root.resolve()),
    )


def main() -> None:
    from findex.index import build_index as canonical_build_index
    from findex.storage import save_index

    parser = argparse.ArgumentParser(description="Build and save a findex index.")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format", choices=("pickle", "json"), default=None)
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Record token positions for phrase search",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    file_format = args.format or ("json" if args.out.suffix == ".json" else "pickle")

    tracemalloc.start()
    start = perf_counter()
    try:
        index = canonical_build_index(args.corpus, with_positions=args.positions)
        built = perf_counter()
        save_index(index, args.out, file_format)
        finished = perf_counter()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"Documents: {index.num_docs}")
    print(f"Vocabulary: {len(index)}")
    print(f"Positions: {index.with_positions}")
    print(f"Format: {file_format}")
    print(f"Saved: {args.out}")
    print(f"Size: {args.out.stat().st_size} bytes")
    print(f"Indexing time: {built - start:.4f} s")
    print(f"Save time: {finished - built:.4f} s")
    print(f"Peak memory: {peak / 1024**2:.4f} MiB")


if __name__ == "__main__":
    main()
