"""Reproducible multi-run indexing benchmark and speedup chart for Lab 05.

Run on the machine being studied; never fabricate timings in the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter, process_time
from typing import TypedDict, cast

from findex.parallel import ExecutorKind, build_parallel


class Measurement(TypedDict):
    wall_s: float
    cpu_s: float
    peak_parent_rss_mib: float
    merge_s: float


class Summary(Measurement):
    python_tag: str
    python_version: str
    gil_enabled: bool
    executor: str
    workers: int
    repeats: int
    speedup: float
    documents: int
    cpu_count: int


COLUMNS = list(Summary.__annotations__)


def _children_cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def _parent_peak_rss_mib() -> float:
    # Each benchmark run is a fresh interpreter. ru_maxrss measures that
    # interpreter's peak, not the combined RSS of spawned worker processes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**2) if sys.platform == "darwin" else raw / 1024


def single_run(
    corpus: Path, executor: ExecutorKind, workers: int, positions: bool
) -> Measurement:
    """Include parent and reaped child CPU, but parent-only peak RSS."""
    cpu_before = process_time() + _children_cpu()
    started = perf_counter()
    result = build_parallel(
        corpus, executor=executor, workers=workers, with_positions=positions
    )
    elapsed = perf_counter() - started
    cpu = process_time() + _children_cpu() - cpu_before
    return {
        "wall_s": elapsed,
        "cpu_s": cpu,
        "peak_parent_rss_mib": _parent_peak_rss_mib(),
        "merge_s": result.merge_seconds,
    }


def _gil_enabled() -> bool:
    checker = getattr(sys, "_is_gil_enabled", None)
    return bool(checker()) if callable(checker) else True


def _run_child(
    corpus: Path,
    executor: ExecutorKind,
    workers: int,
    positions: bool,
    expect_no_gil: bool,
) -> Measurement:
    args = [
        sys.executable,
        "-m",
        "findex.bench_parallel",
        "--one",
        "--corpus",
        str(corpus),
        "--executor",
        executor,
        "--workers",
        str(workers),
    ]
    if positions:
        args.append("--positions")
    if expect_no_gil:
        args.append("--expect-no-gil")
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Benchmark returned invalid data: {completed.stdout!r}"
        ) from exc
    return cast(Measurement, value)


def _save_chart(rows: list[Summary], target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    groups: dict[tuple[str, str], list[Summary]] = {}
    for row in rows:
        if row["executor"] != "serial":
            groups.setdefault((row["python_tag"], row["executor"]), []).append(row)
    max_workers = max((row["workers"] for row in rows), default=1)
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        [1, max_workers], [1, max_workers], "--", color="gray", label="Ideal linear"
    )
    for (tag, executor), records in sorted(groups.items()):
        records.sort(key=lambda record: record["workers"])
        ax.plot(  # pyright: ignore[reportUnknownMemberType]
            [record["workers"] for record in records],
            [record["speedup"] for record in records],
            marker="o",
            label=f"{tag}: {executor}",
        )
    ax.set_xlabel("Workers")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Speedup vs same interpreter serial (x)")  # pyright: ignore[reportUnknownMemberType]
    ax.set_title("Findex parallel indexing — measured on local machine")  # pyright: ignore[reportUnknownMemberType]
    ax.grid(True, alpha=0.3)  # pyright: ignore[reportUnknownMemberType]
    ax.legend()  # pyright: ignore[reportUnknownMemberType]
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=str, default="1,2,4,8,cpu")
    parser.add_argument("--modes", type=str, default="serial,threads,processes")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--positions", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=Path("docs/lab05_metrics.csv"))
    parser.add_argument(
        "--plot", type=Path, default=Path("docs/images/lab05_speedup.png")
    )
    parser.add_argument("--label", type=str, default="cpython-3.13")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--expect-no-gil",
        action="store_true",
        help="Fail if GIL is active (3.13t verification).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Useful in the free-threaded environment.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw a chart from existing CSV measurements.",
    )
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--executor", choices=("serial", "threads", "processes"), default="serial"
    )
    args = parser.parse_args()
    if not args.plot_only and not args.corpus.is_dir():
        parser.error("--corpus must be an existing directory")
    if args.repeat < 3 and not args.one and not args.plot_only:
        parser.error("At least three repetitions are required")
    if args.expect_no_gil and _gil_enabled():
        parser.error(
            "GIL is enabled; use a free-threaded interpreter with GIL disabled"
        )
    if args.one:
        executor = cast(ExecutorKind, args.executor)
        sample = single_run(args.corpus, executor, int(args.workers), args.positions)
        if args.expect_no_gil and _gil_enabled():
            parser.error("GIL became enabled during this benchmark")
        print(json.dumps(sample))
        return
    count = os.cpu_count() or 1
    worker_counts = sorted(
        {
            count if value.strip() == "cpu" else int(value)
            for value in args.workers.split(",")
        }
    )
    if any(value <= 0 for value in worker_counts):
        parser.error("Worker count must be positive")
    modes = [value.strip() for value in args.modes.split(",")]
    if any(value not in ("serial", "threads", "processes") for value in modes):
        parser.error("Invalid --modes")
    if "serial" not in modes and not args.append:
        parser.error("A baseline serial run is required")
    rows: list[Summary] = []
    if (args.append or args.plot_only) and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as file:
            for item in csv.DictReader(file):
                rows.append(
                    {
                        "python_tag": item["python_tag"],
                        "python_version": item["python_version"],
                        "gil_enabled": item["gil_enabled"] == "True",
                        "executor": item["executor"],
                        "workers": int(item["workers"]),
                        "repeats": int(item["repeats"]),
                        "speedup": float(item["speedup"]),
                        "documents": int(item["documents"]),
                        "cpu_count": int(item["cpu_count"]),
                        "wall_s": float(item["wall_s"]),
                        "cpu_s": float(item["cpu_s"]),
                        "peak_parent_rss_mib": float(item["peak_parent_rss_mib"]),
                        "merge_s": float(item["merge_s"]),
                    }
                )
    if args.plot_only:
        if not rows:
            parser.error("The CSV does not exist or has no measurements")
        _save_chart(rows, args.plot)
        print(f"Saved {args.plot}")
        return
    corpus = args.corpus.resolve()
    docs = sum(1 for path in corpus.rglob("*.txt") if path.is_file())
    print(
        f"Python {platform.python_version()}; GIL enabled: {_gil_enabled()}",
        file=sys.stderr,
    )
    baseline: float | None = None
    jobs: list[tuple[ExecutorKind, int]] = []
    if "serial" in modes:
        jobs.append(("serial", 1))
    for mode in modes:
        if mode != "serial":
            jobs.extend((cast(ExecutorKind, mode), n) for n in worker_counts)
    for executor, workers in jobs:
        samples: list[Measurement] = []
        for repetition in range(args.repeat):
            sample = _run_child(
                corpus, executor, workers, args.positions, args.expect_no_gil
            )
            samples.append(sample)
            print(
                f"{args.label} {executor} {workers} run {repetition + 1}/{args.repeat}: {sample['wall_s']:.4f}s",
                file=sys.stderr,
            )
        med = {
            key: median(sample[key] for sample in samples)
            for key in ("wall_s", "cpu_s", "peak_parent_rss_mib", "merge_s")
        }
        if executor == "serial":
            baseline = med["wall_s"]
        if baseline is None:
            match = next(
                (
                    row
                    for row in rows
                    if row["python_tag"] == args.label and row["executor"] == "serial"
                ),
                None,
            )
            if match is None:
                raise ValueError("No serial baseline for this interpreter label")
            baseline = match["wall_s"]
        rows.append(
            {
                "python_tag": args.label,
                "python_version": platform.python_version(),
                "gil_enabled": _gil_enabled(),
                "executor": executor,
                "workers": workers,
                "repeats": args.repeat,
                "speedup": baseline / med["wall_s"] if med["wall_s"] > 0 else 0.0,
                "documents": docs,
                "cpu_count": count,
                "wall_s": med["wall_s"],
                "cpu_s": med["cpu_s"],
                "peak_parent_rss_mib": med["peak_parent_rss_mib"],
                "merge_s": med["merge_s"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if not args.skip_plot:
        _save_chart(rows, args.plot)
    print(f"Saved {args.output}" + ("" if args.skip_plot else f" and {args.plot}"))
    for row in rows:
        print(
            f"{row['python_tag']:16} {row['executor']:9} {row['workers']:2} "
            f"wall={row['wall_s']:.4f}s cpu={row['cpu_s']:.4f}s "
            f"rss={row['peak_parent_rss_mib']:.1f}MiB merge={row['merge_s']:.4f}s "
            f"speedup={row['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
