import subprocess
import sys
from pathlib import Path


def test_cli_phrase_and_cache(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("python event loop", encoding="utf-8")
    (corpus / "b.txt").write_text("python event driven loop", encoding="utf-8")
    index_path = tmp_path / "search.json"
    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "findex.index",
            str(corpus),
            "--out",
            str(index_path),
            "--positions",
            "--verbose",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert "Documents: 2" in created.stdout
    assert "Positions: True" in created.stdout
    assert "TIMED build_index" in created.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "findex.ranking",
            str(index_path),
            'python AND "event loop"',
            "--top",
            "5",
            "--repeat",
            "2",
            "--verbose",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert "score=" in result.stdout
    assert "a.txt" in result.stdout
    assert "b.txt" not in result.stdout
    assert "CACHE HIT" in result.stderr
    assert "TIMED load_index" in result.stderr
    assert "TIMED RankedSearcher.search" in result.stderr
