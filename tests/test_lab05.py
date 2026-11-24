"""Index equivalence, stable document IDs, spawn and failing workers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Chunk, build_index, build_partial, merge
from findex.parallel import ExecutorKind, build_parallel, make_chunks
from findex.storage import load_index


@pytest.fixture
def lab05_corpus(tmp_path: Path) -> Path:
    (tmp_path / "b.txt").write_text("hello world world", encoding="utf-8")
    (tmp_path / "a.txt").write_text("hello python", encoding="utf-8")
    (tmp_path / "c.txt").write_text("hello python world", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("positions", [False, True])
def test_single_partial_equals_lab04(lab05_corpus: Path, positions: bool) -> None:
    original = build_index(lab05_corpus, with_positions=positions)
    chunk = make_chunks(lab05_corpus, 1, positions)[0]
    assert merge([build_partial(chunk)]) == original


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("positions", [False, True])
def test_parallel_equivalence(
    lab05_corpus: Path, executor: str, workers: int, positions: bool
) -> None:
    original = build_index(lab05_corpus, with_positions=positions)
    result = build_parallel(
        lab05_corpus,
        executor=cast(ExecutorKind, executor),
        workers=workers,
        with_positions=positions,
    )
    assert result.index == original
    assert sorted(result.index.documents) == [0, 1, 2]
    for values in result.index.postings.values():
        assert [posting.doc_id for posting in values] == sorted(
            {posting.doc_id for posting in values}
        )


def test_duplicate_ids_rejected(lab05_corpus: Path) -> None:
    partial = build_partial(make_chunks(lab05_corpus, 1, True)[0])
    with pytest.raises(ValueError, match="Duplicate doc_id"):
        merge([partial, partial])


def test_bad_workers(lab05_corpus: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_chunks(lab05_corpus, 0, False)


def test_spawn_worker_exception_propagates(lab05_corpus: Path) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    bad = Chunk(str(lab05_corpus), (str(lab05_corpus / "absent.txt"),), 0)
    with (
        ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
        pytest.raises(FileNotFoundError),
    ):
        pool.submit(build_partial, bad).result(timeout=30)


def test_cli_modes_and_roundtrip(lab05_corpus: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    expected = build_index(lab05_corpus, with_positions=True)
    for mode in ("serial", "threads", "processes"):
        output = tmp_path / f"{mode}.json"
        result = runner.invoke(
            app,
            [
                "index",
                str(lab05_corpus),
                "--out",
                str(output),
                "--executor",
                mode,
                "--workers",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert load_index(output, "json") == expected


def test_cli_bad_workers(lab05_corpus: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "index",
            str(lab05_corpus),
            "--out",
            str(tmp_path / "x.bin"),
            "--workers",
            "0",
        ],
    )
    assert result.exit_code != 0
