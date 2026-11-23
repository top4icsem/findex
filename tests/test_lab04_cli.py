"""CLI integration tests using Typer's test runner."""

import json
from pathlib import Path

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("index", "search", "stats"):
        assert command in result.output


def test_index_search_and_json(sample_corpus: Path, tmp_path: Path) -> None:
    dest = tmp_path / "index.json"
    created = runner.invoke(app, ["index", str(sample_corpus), "--out", str(dest)])
    assert created.exit_code == 0, created.output
    assert dest.is_file()

    shown = runner.invoke(app, ["search", str(dest), "python", "--top", "2"])
    assert shown.exit_code == 0, shown.output
    assert "a.txt" in shown.output or "b.txt" in shown.output

    raw = runner.invoke(app, ["search", str(dest), '"event loop"', "--json"])
    assert raw.exit_code == 0, raw.output
    results = json.loads(raw.stdout)
    assert len(results) == 1
    assert results[0]["path"] == "a.txt"
    assert "score" in results[0]


def test_stats(sample_corpus: Path) -> None:
    result = runner.invoke(app, ["stats", str(sample_corpus)])
    assert result.exit_code == 0, result.output
    assert "Documents" in result.output
    assert "3" in result.output


def test_expected_errors(sample_corpus: Path, tmp_path: Path) -> None:
    missing = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert missing.exit_code != 0
    assert "Error:" in missing.output
    assert "Traceback" not in missing.output

    dest = tmp_path / "index.bin"
    assert (
        runner.invoke(app, ["index", str(sample_corpus), "--out", str(dest)]).exit_code
        == 0
    )
    invalid = runner.invoke(app, ["search", str(dest), "python AND"])
    assert invalid.exit_code != 0
    assert "Error:" in invalid.output
    assert "Traceback" not in invalid.output
