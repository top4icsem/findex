"""Reproducible 200-page crawler benchmark on a self-hosted loopback fixture."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import psutil

from findex.crawler import CrawlConfig, CrawlStats, crawl
from findex.fixture_site import start_local_site

COLUMNS: list[str] = [
    "concurrency",
    "max_pages",
    "wall_s",
    "pages_per_s",
    "errors",
    "peak_rss_mib",
    "repeats",
]


async def one_run(
    origin: str, max_pages: int, concurrency: int, host_delay: float
) -> dict[str, float]:
    config = CrawlConfig(
        seed=origin + "/page/000",
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=concurrency,
        min_host_delay=host_delay,
        user_agent=f"FindexLabCrawler/0.6 (+{origin}/contact)",
    )
    stats = CrawlStats()
    process = psutil.Process(os.getpid())
    peak = float(process.memory_info().rss)

    def update(_stats: CrawlStats) -> None:
        nonlocal peak
        peak = max(peak, float(process.memory_info().rss))

    first = perf_counter()

    def log_attempt(
        url: str,
        status: int,
        size: int,
        elapsed: float,
        attempt: int,
        error: str | None,
    ) -> None:
        logging.getLogger("findex.crawl").info(
            "URL=%s status=%d bytes=%d time_s=%.4f attempt=%d error=%s",
            url,
            status,
            size,
            elapsed,
            attempt,
            error or "none",
        )

    async for _ in crawl(
        config, stats=stats, on_progress=update, on_attempt=log_attempt
    ):
        pass
    wall = perf_counter() - first
    if stats.pages != max_pages:
        raise ValueError(f"Expected {max_pages} pages; crawled {stats.pages}")
    return {
        "wall_s": wall,
        "pages_per_s": stats.pages / wall,
        "errors": float(stats.errors),
        "peak_rss_mib": peak / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--concurrency", default="1,5,20")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--response-delay", type=float, default=0.03)
    parser.add_argument("--host-delay", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=Path("docs/lab06_metrics.csv"))
    parser.add_argument(
        "--failure-log", type=Path, default=Path("docs/lab06_failure.log")
    )
    args = parser.parse_args()
    if args.max_pages < 200 or args.repeat < 1:
        parser.error("Use at least 200 pages and at least one repeat")
    counts = [int(value) for value in args.concurrency.split(",")]
    if any(count < 1 for count in counts):
        parser.error("All concurrency values must be positive")
    rows: list[dict[str, float | int]] = []
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    with args.failure_log.open("w", encoding="utf-8") as logfile:
        logfile.write("Local loopback test: owned website; robots allow crawl.\n")
        for concurrency in counts:
            samples: list[dict[str, float]] = []
            for repeat in range(1, args.repeat + 1):
                server, thread, origin = start_local_site(args.response_delay)
                handler = logging.StreamHandler(logfile)
                handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
                logger = logging.getLogger("findex.crawl")
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)
                try:
                    value = asyncio.run(
                        one_run(origin, args.max_pages, concurrency, args.host_delay)
                    )
                    samples.append(value)
                    print(
                        f"concurrency={concurrency} run={repeat}/{args.repeat}: "
                        f"{value['wall_s']:.3f}s pages={args.max_pages}",
                        flush=True,
                    )
                finally:
                    logger.removeHandler(handler)
                    handler.flush()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)
            med = {
                key: median(sample[key] for sample in samples)
                for key in ("wall_s", "pages_per_s", "errors", "peak_rss_mib")
            }
            rows.append(
                {
                    "concurrency": concurrency,
                    "max_pages": args.max_pages,
                    "repeats": args.repeat,
                    **med,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="\n", encoding="utf-8") as file:
        writer: csv.DictWriter[str] = csv.DictWriter(file, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output} and {args.failure_log}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
