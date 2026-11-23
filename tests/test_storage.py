import json
from pathlib import Path

import pytest

from findex.index import InvertedIndex, build_index
from findex.search import search
from findex.storage import load_index, save_index


@pytest.fixture
def sample_index(tmp_path: Path) -> InvertedIndex:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "a.txt").write_text("python search python", encoding="utf-8")
    (corpus / "b.txt").write_text("python memory", encoding="utf-8")
    (corpus / "c.txt").write_text("search engine", encoding="utf-8")

    return build_index(corpus)


@pytest.mark.parametrize(
    ("format_name", "extension"),
    [
        ("pickle", ".bin"),
        ("json", ".json"),
    ],
)
def test_roundtrip(
    sample_index: InvertedIndex,
    tmp_path: Path,
    format_name: str,
    extension: str,
):
    path = tmp_path / f"index{extension}"

    save_index(sample_index, path, format_name)

    assert path.exists()
    assert path.stat().st_size > 0

    loaded = load_index(path, format_name)

    assert loaded == sample_index
    assert search(loaded, "python AND search") == [0]
    assert search(loaded, "python OR search") == [0, 1, 2]


def test_bad_save_format(
    sample_index: InvertedIndex,
    tmp_path: Path,
):
    with pytest.raises(ValueError):
        save_index(sample_index, tmp_path / "bad.xyz", "bad")


def test_bad_load_format(tmp_path: Path):
    with pytest.raises(ValueError):
        load_index(tmp_path / "bad.xyz", "bad")


def test_bad_json_version(tmp_path: Path):
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps({"version": 999}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_index(path, "json")


@pytest.mark.parametrize("format_name", ["pickle", "json"])
def test_missing_file(tmp_path: Path, format_name: str):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "missing.bin", format_name)
