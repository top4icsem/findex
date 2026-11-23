import json
import pickle
from dataclasses import asdict
from pathlib import Path

from findex.index import DocMeta, InvertedIndex, Posting


def save_index(
    index: InvertedIndex,
    path: Path,
    format: str = "pickle",
) -> None:
    """Save an inverted index in Pickle or JSON format."""
    if format == "pickle":
        with path.open("wb") as file:
            pickle.dump(index, file, protocol=pickle.HIGHEST_PROTOCOL)

    elif format == "json":
        data = {
            "version": 1,
            "postings": {
                term: [[posting.doc_id, posting.frequency] for posting in posting_list]
                for term, posting_list in index.postings.items()
            },
            "documents": [asdict(meta) for meta in index.documents.values()],
        }

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_index(
    path: Path,
    format: str = "pickle",
) -> InvertedIndex:
    """Load an inverted index from a trusted local file."""
    if format == "pickle":
        # WARNING: Never unpickle files from untrusted sources.
        with path.open("rb") as file:
            index = pickle.load(file)

        if not isinstance(index, InvertedIndex):
            raise ValueError("Invalid index file")

        return index

    if format == "json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if data.get("version") != 1:
            raise ValueError("Unsupported index version")

        postings = {
            term: [
                Posting(doc_id=int(doc_id), frequency=int(freq))
                for doc_id, freq in posting_list
            ]
            for term, posting_list in data["postings"].items()
        }

        documents = {item["doc_id"]: DocMeta(**item) for item in data["documents"]}

        return InvertedIndex(
            postings=postings,
            documents=documents,
        )

    raise ValueError(f"Unsupported format: {format}")
