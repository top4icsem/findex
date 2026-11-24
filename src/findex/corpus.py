import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Read text documents one by one using a generator."""
    if root.is_file() and root.suffix.lower() == ".jsonl":
        with root.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                raw: object = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError(f"Invalid JSONL record on line {line_number}")
                record = cast("dict[str, object]", raw)
                url = record.get("url")
                text = record.get("text")
                if not isinstance(url, str) or not isinstance(text, str):
                    raise TypeError(f"Missing url/text on line {line_number}")
                yield Document(doc_id=url, path=root, text=text)
        return

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
