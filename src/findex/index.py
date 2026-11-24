"""The inverted index, Python data model methods, and indexing CLI."""

import argparse
import logging
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from time import perf_counter

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


@dataclass(frozen=True, slots=True)
class Chunk:
    """Picklable file-only task with a reserved range of document IDs."""

    root: str
    paths: tuple[str, ...]
    start_doc_id: int
    with_positions: bool = False


def discover_paths(root: Path) -> list[Path]:
    """Use exactly the ordering of the original corpus generator."""
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {root}")
    return [path for path in sorted(root.rglob("*.txt")) if path.is_file()]


def build_partial(
    chunk: Chunk,
    progress: Callable[[int], None] | None = None,
) -> Index:
    """Build one independent partial index; callable from a spawned process.

    Inputs contain paths, never document contents. No shared mutable structures.
    """
    root = Path(chunk.root)
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}
    for offset, name in enumerate(chunk.paths):
        path = Path(name)
        text = path.read_text(encoding="utf-8", errors="replace")
        doc_id = chunk.start_doc_id + offset
        if chunk.with_positions:
            positions_by_term: defaultdict[str, list[int]] = defaultdict(list)
            for position, term in enumerate(tokenize(text)):
                positions_by_term[term].append(position)
            frequencies: dict[str, int] = {
                term: len(positions) for term, positions in positions_by_term.items()
            }
        else:
            positions_by_term = defaultdict(list)
            frequencies = dict(Counter(tokenize(text)))
        documents[doc_id] = DocMeta(
            doc_id=doc_id,
            path=path.relative_to(root).as_posix(),
            length=sum(frequencies.values()),
            preview=text[:200],
        )
        for term, frequency in frequencies.items():
            postings[term].append(
                Posting(
                    doc_id=doc_id,
                    frequency=frequency,
                    positions=tuple(positions_by_term[term])
                    if chunk.with_positions
                    else (),
                )
            )
        if progress is not None:
            progress(offset + 1)
    return Index(
        postings=dict(postings),
        documents=documents,
        with_positions=chunk.with_positions,
        corpus_root=str(root.resolve()),
    )


def merge(partials: list[Index]) -> Index:
    """Combine disjoint doc-ID chunks into one deterministic, sorted index."""
    if not partials:
        raise ValueError("Expected at least one partial index")
    with_positions = partials[0].with_positions
    corpus_root = partials[0].corpus_root
    documents: dict[int, DocMeta] = {}
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    for partial in partials:
        if (
            partial.with_positions != with_positions
            or partial.corpus_root != corpus_root
        ):
            raise ValueError("Incompatible partial indexes")
        for doc_id, meta in partial.documents.items():
            if doc_id in documents:
                raise ValueError(f"Duplicate doc_id: {doc_id}")
            documents[doc_id] = meta
        for term, values in partial.postings.items():
            postings[term].extend(values)
    for term, values in postings.items():
        values.sort(key=lambda value: value.doc_id)
        if any(a.doc_id == b.doc_id for a, b in pairwise(values)):
            raise ValueError(f"Duplicate posting for term: {term}")
    return Index(
        postings=dict(postings),
        documents=dict(sorted(documents.items())),
        with_positions=with_positions,
        corpus_root=corpus_root,
    )


@timed
def build_index(
    root: Path,
    with_positions: bool = False,
    progress: Callable[[int], None] | None = None,
) -> Index:
    """Lab 01-04 compatible sequential entry point, now using partial/merge."""
    root = Path(root).resolve()
    paths = discover_paths(root)
    task = Chunk(str(root), tuple(str(path) for path in paths), 0, with_positions)
    return merge([build_partial(task, progress=progress)])


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
