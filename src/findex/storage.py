"""Trusted Pickle or interoperable JSON index persistence."""

import json
import pickle
from dataclasses import asdict
from pathlib import Path

from findex.index import DocMeta, InvertedIndex, Posting
from findex.timing import timed


def save_index(index: InvertedIndex, path: Path, format: str = "pickle") -> None:
    """Save index, including optional term positions."""
    if format == "pickle":
        with path.open("wb") as file:
            pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if format == "json":
        data = {
            "version": 2,
            "with_positions": index.with_positions,
            "corpus_root": index.corpus_root,
            "postings": {
                term: [
                    [p.doc_id, p.frequency, list(p.positions)]
                    if index.with_positions
                    else [p.doc_id, p.frequency]
                    for p in posting_list
                ]
                for term, posting_list in index.postings.items()
            },
            "documents": [asdict(meta) for meta in index.documents.values()],
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, separators=(",", ":"))
        return

    raise ValueError(f"Unsupported format: {format}")


@timed
def load_index(path: Path, format: str = "pickle") -> InvertedIndex:
    """Load only trusted Pickle files; never unpickle untrusted content."""
    if format == "pickle":
        with path.open("rb") as file:
            index = pickle.load(file)
        if not isinstance(index, InvertedIndex):
            raise ValueError("Invalid index file")
        return index

    if format == "json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if data.get("version") not in (1, 2):
            raise ValueError("Unsupported index version")
        postings: dict[str, list[Posting]] = {}
        for term, raw_list in data["postings"].items():
            postings[term] = [
                Posting(
                    doc_id=int(item[0]),
                    frequency=int(item[1]),
                    positions=tuple(int(pos) for pos in item[2])
                    if len(item) == 3
                    else (),
                )
                for item in raw_list
            ]
        documents: dict[int, DocMeta] = {
            meta["doc_id"]: DocMeta(**meta) for meta in data["documents"]
        }
        return InvertedIndex(
            postings=postings,
            documents=documents,
            with_positions=bool(data.get("with_positions", False)),
            corpus_root=str(data.get("corpus_root", "")),
        )

    raise ValueError(f"Unsupported format: {format}")
