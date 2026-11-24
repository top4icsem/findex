"""Findex command-line interface: build, search and corpus statistics."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import sys
import traceback
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, NoReturn
from urllib.parse import urlsplit

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawl_index import build_jsonl_index
from findex.crawler import CrawlConfig, CrawlStats, canonicalize, crawl, origin_of
from findex.open_index import open_index
from findex.parallel import ExecutorKind, build_parallel
from findex.ranking import RankedSearcher
from findex.scoring import BM25, Scorer, Tfidf
from findex.stats import collect_stats
from findex.storage import save_index

app = typer.Typer(
    help="Findex: index and search a local text corpus.", no_args_is_help=True
)
console = Console()
progress_console = Console(stderr=True)

ScorerName = Literal["bm25", "tfidf"]
StorageName = Literal["pickle", "json"]


class ExecutorChoice(str, Enum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


class ScorerChoice(str, Enum):
    bm25 = "bm25"
    tfidf = "tfidf"


@app.callback()
def configure_logging(
    verbose: Annotated[
        int,
        typer.Option(
            "-v", "--verbose", count=True, help="Repeat to increase logging verbosity."
        ),
    ] = 0,
) -> None:
    """Configure diagnostics once. Only results go to stdout."""
    level = (
        logging.DEBUG if verbose >= 2 else logging.INFO if verbose else logging.WARNING
    )
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def fail(message: str) -> NoReturn:
    """Report an expected user error without exposing a stack trace."""
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _format_for(path: Path) -> StorageName:
    return "json" if path.suffix.lower() == ".json" else "pickle"


def _make_scorer(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else Tfidf()


@app.command("index")
def index_command(
    corpus: Annotated[
        Path, typer.Argument(help="Directory containing .txt documents.")
    ],
    out: Annotated[
        Path, typer.Option("--out", help="Index destination (.bin or .json).")
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option(
            "--positions/--no-positions",
            help="Store token offsets for exact phrase searches.",
        ),
    ] = True,
    workers: Annotated[int, typer.Option("--workers", min=1, help="Worker count.")] = 1,
    executor: Annotated[
        ExecutorChoice,
        typer.Option("--executor", help="serial, threads or processes."),
    ] = ExecutorChoice.processes,
) -> None:
    """Index text documents and save the index."""
    if not corpus.is_dir() and not (corpus.is_file() and corpus.suffix == ".jsonl"):
        fail(f"Corpus directory or JSONL file not found: {corpus}")
    if out.suffix.lower() not in (".bin", ".json"):
        fail("Index output must end in .bin or .json")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        jsonl = corpus.is_file()
        total = 0 if jsonl else sum(1 for p in corpus.rglob("*.txt") if p.is_file())
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=progress_console,
        ) as progress:
            task = progress.add_task("Indexing", total=max(1, total))

            def on_document(done: int) -> None:
                progress.update(task, completed=min(done, max(1, total)))

            mode: ExecutorKind = (
                "serial"
                if executor is ExecutorChoice.serial
                else "threads"
                if executor is ExecutorChoice.threads
                else "processes"
            )
            if jsonl:
                # JSONL indexing runs outside the asyncio event loop.
                index = build_jsonl_index(corpus, with_positions=positions)
            else:
                result = build_parallel(
                    corpus,
                    executor=mode,
                    workers=workers,
                    with_positions=positions,
                    progress=on_document,
                )
                index = result.index
            progress.update(task, completed=max(1, total))
        save_index(index, out, _format_for(out))
    except (OSError, UnicodeError, ValueError) as exc:
        if executor is not ExecutorChoice.serial:
            # Worker failures must expose their (possibly remote) traceback.
            traceback.print_exception(exc, file=sys.stderr)
            raise typer.Exit(code=1) from None
        fail(str(exc))
    console.print(f"Indexed {index.num_docs} documents, {len(index)} terms -> {out}")


def _is_local(url: str) -> bool:
    host = urlsplit(url).hostname
    if host == "localhost":
        return True
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.command("crawl")
def crawl_command(
    seed: Annotated[str, typer.Argument(help="Initial permitted HTTP URL.")],
    out: Annotated[
        Path, typer.Option("--out", help="Streaming JSONL destination.")
    ] = Path("crawl/pages.jsonl"),
    max_pages: Annotated[int, typer.Option("--max-pages", min=1)] = 200,
    concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 5,
    per_host: Annotated[int, typer.Option("--per-host", min=1)] = 5,
    host_delay: Annotated[float, typer.Option("--host-delay", min=0)] = 0.02,
    contact: Annotated[
        str,
        typer.Option(
            "--contact",
            help="Actual contact e-mail or URL; required for external sites.",
        ),
    ] = "",
    log_file: Annotated[
        Path, typer.Option("--log", help="Per-request status log.")
    ] = Path("crawl.log"),
    timeout: Annotated[float, typer.Option("--timeout", min=0.01)] = 8.0,
) -> None:
    """Crawl only an authorized origin, writing each result to JSONL immediately."""
    try:
        clean_seed = canonicalize(seed)
        if not _is_local(clean_seed) and not contact.strip():
            fail(
                "External sites require a real --contact address and permission to crawl."
            )
        actual_contact = contact.strip() or f"{origin_of(clean_seed)}/contact"
        if not (
            actual_contact.startswith(("https://", "http://", "mailto:"))
            or "@" in actual_contact
        ):
            fail("--contact must be a contact URL or e-mail address")
        user_agent = f"FindexLabCrawler/0.6 (+{actual_contact})"
        config = CrawlConfig(
            seed=clean_seed,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_host_delay=host_delay,
            timeout_s=timeout,
            user_agent=user_agent,
            allowed_origins=frozenset({origin_of(clean_seed)}),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        crawl_logger = logging.getLogger("findex.crawl")
        crawl_logger.setLevel(logging.INFO)
        crawl_logger.addHandler(file_handler)

        def on_attempt(
            url: str,
            status: int,
            size: int,
            elapsed: float,
            attempt: int,
            error: str | None,
        ) -> None:
            crawl_logger.info(
                "URL=%s status=%s bytes=%d time_s=%.4f attempt=%d error=%s",
                url,
                status,
                size,
                elapsed,
                attempt,
                error or "none",
            )

        stats = CrawlStats()

        async def run() -> None:
            with (
                out.open("w", encoding="utf-8", newline="\n") as stream,
                Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    console=progress_console,
                ) as visual,
            ):
                task = visual.add_task("Crawling", total=max_pages)

                def notify(snapshot: CrawlStats) -> None:
                    visual.update(
                        task,
                        completed=snapshot.pages,
                        description=(
                            f"Crawl {snapshot.pages_per_second:.1f} pages/s "
                            f"active={snapshot.in_flight} err={snapshot.errors} queue={snapshot.queued}"
                        ),
                    )

                async for page in crawl(
                    config, stats=stats, on_progress=notify, on_attempt=on_attempt
                ):
                    stream.write(
                        json.dumps(page.as_record(), ensure_ascii=False) + "\n"
                    )
                    stream.flush()

        try:
            asyncio.run(run())
        finally:
            crawl_logger.removeHandler(file_handler)
            file_handler.close()
        console.print(f"Crawled {stats.pages} pages, {stats.errors} errors -> {out}")
        console.print(f"Request log: {log_file}")
    except (OSError, ValueError, UnicodeError) as exc:
        fail(str(exc))


@app.command("search")
def search_command(
    index_file: Annotated[Path, typer.Argument(help="Saved .bin or .json index.")],
    query: Annotated[
        str, typer.Argument(help="Boolean query, with optional quoted phrases.")
    ],
    scorer: Annotated[
        ScorerChoice, typer.Option("--scorer", help="Ranking model.")
    ] = ScorerChoice.bm25,
    top: Annotated[int, typer.Option("--top", help="Maximum number of matches.")] = 5,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output machine-readable JSON only.")
    ] = False,
) -> None:
    """Search the indexed corpus and rank results."""
    if not index_file.is_file():
        fail(f"Index file not found: {index_file}")
    if top < 0:
        fail("--top must not be negative")
    scorer_name: ScorerName = "bm25" if scorer is ScorerChoice.bm25 else "tfidf"
    try:
        with open_index(index_file) as index:
            results = RankedSearcher(index, _make_scorer(scorer_name)).search(
                query, top_k=top
            )
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        fail(str(exc))

    rows = [
        {
            "doc_id": r.doc_id,
            "score": round(r.score, 9),
            "path": r.path,
            "snippet": r.snippet,
        }
        for r in results
    ]
    if json_output:
        typer.echo(json.dumps(rows, ensure_ascii=False))
        return
    table = Table(title=f"Findex results ({scorer_name})")
    table.add_column("#", justify="right")
    table.add_column("Document")
    table.add_column("Score", justify="right")
    table.add_column("Snippet", overflow="fold")
    for pos, row in enumerate(rows, 1):
        table.add_row(
            str(pos), str(row["path"]), f"{row['score']:.6f}", str(row["snippet"])
        )
    console.print(table)


@app.command("stats")
def stats_command(
    corpus: Annotated[
        Path, typer.Argument(help="Directory containing .txt documents.")
    ],
) -> None:
    """Compute one-pass text corpus statistics."""
    if not corpus.is_dir():
        fail(f"Corpus directory not found: {corpus}")
    try:
        stats = collect_stats(corpus)
    except (OSError, UnicodeError, ValueError) as exc:
        fail(str(exc))
    table = Table(title="Findex corpus statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in (
        ("Documents", str(stats.documents)),
        ("Tokens", str(stats.tokens)),
        ("Vocabulary", str(stats.vocabulary)),
        ("Time (s)", f"{stats.elapsed_seconds:.6f}"),
        ("Peak memory (MiB)", f"{stats.peak_memory_mb:.4f}"),
    ):
        table.add_row(key, value)
    console.print(table)


def main() -> None:
    """Executable entry point for `python -m findex.cli`."""
    app()


if __name__ == "__main__":
    main()
