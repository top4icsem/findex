import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("extension", ["bin", "json"])
def test_cli_roundtrip(tmp_path: Path, extension: str):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "a.txt").write_text("python search", encoding="utf-8")
    (corpus / "b.txt").write_text("python memory", encoding="utf-8")

    output = tmp_path / f"index.{extension}"

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "findex.index",
            str(corpus),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Documents: 2" in build.stdout
    assert output.exists()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "findex.search",
            str(output),
            "python AND search",
            "--engine",
            "merge",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Matches: 1" in result.stdout
    assert "a.txt" in result.stdout
