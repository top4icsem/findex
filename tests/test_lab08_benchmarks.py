"""Microbenchmark fixtures collected by pytest-benchmark in GitHub Actions.

Use `uv run pytest --benchmark-only -q` for a focused performance report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from findex.index import build_index
from findex.numpy_rank import NumpyRanker

pytest.importorskip("pytest_benchmark")


def test_benchmark_search(benchmark: Any, sample_index: Any) -> None:
    ranker = NumpyRanker(sample_index)
    benchmark(ranker.search, "python", 5)


def test_benchmark_build_index(benchmark: Any, sample_corpus: Path) -> None:
    benchmark(build_index, sample_corpus, with_positions=True)
