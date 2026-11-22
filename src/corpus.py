
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Read text documents one by one using a generator."""
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {root}")

    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8", errors="replace")

        yield Document(
            doc_id=path.relative_to(root).as_posix(),
            path=path,
            text=text,
        )
