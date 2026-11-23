"""Context manager to safely release the in-memory index."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from findex.index import Index
from findex.storage import load_index


@contextmanager
def open_index(path: Path, format: str | None = None) -> Generator[Index]:
    """Load a trusted index and release its data, also on exceptions."""
    file_format = format or ("json" if path.suffix == ".json" else "pickle")
    index = load_index(path, file_format)
    try:
        yield index
    finally:
        index.close()
