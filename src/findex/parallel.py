"""Parallel, deterministic indexing using picklable file-path tasks."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from findex.index import Chunk, Index, build_partial, discover_paths, merge

type ExecutorKind = Literal["serial", "threads", "processes"]


@dataclass(frozen=True, slots=True)
class BuildResult:
    index: Index
    merge_seconds: float
    chunks: int


def make_chunks(root: Path, workers: int, with_positions: bool) -> list[Chunk]:
    """Reserve contiguous, globally unique IDs before any worker starts."""
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    root = root.resolve()
    paths = discover_paths(root)
    if not paths:
        return [Chunk(str(root), (), 0, with_positions)]
    count = min(workers, len(paths))
    base, extra = divmod(len(paths), count)
    chunks: list[Chunk] = []
    start = 0
    for number in range(count):
        length = base + int(number < extra)
        segment = paths[start : start + length]
        chunks.append(
            Chunk(
                str(root), tuple(str(path) for path in segment), start, with_positions
            )
        )
        start += length
    return chunks


def build_parallel(
    root: Path,
    *,
    executor: ExecutorKind = "serial",
    workers: int = 1,
    with_positions: bool = False,
    progress: Callable[[int], None] | None = None,
) -> BuildResult:
    """Build independently and merge. Failed futures raise with remote traceback."""
    if executor not in {"serial", "threads", "processes"}:
        raise ValueError(f"Unsupported executor: {executor}")
    chunks = make_chunks(Path(root), workers, with_positions)
    partials: list[Index] = []
    done = 0
    if executor == "serial":
        # Exactly one chunk: no extra partition cost for the baseline.
        single = Chunk(
            chunks[0].root,
            tuple(path for chunk in chunks for path in chunk.paths),
            0,
            with_positions,
        )
        chunks = [single]
        partials.append(build_partial(single, progress=progress))
    else:
        pool: Executor
        if executor == "threads":
            pool = ThreadPoolExecutor(max_workers=workers)
        else:
            pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        with pool:
            for chunk, partial in zip(
                chunks, pool.map(build_partial, chunks), strict=True
            ):
                partials.append(partial)
                done += len(chunk.paths)
                if progress is not None:
                    progress(done)
    start = perf_counter()
    output = merge(partials)
    return BuildResult(output, perf_counter() - start, len(chunks))
