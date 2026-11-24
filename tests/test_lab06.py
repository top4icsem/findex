"""Lab 06 tests: HTTP requests are mocked; no live network required."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawl_index import build_jsonl_index
from findex.crawler import (
    CrawlConfig,
    CrawlStats,
    RequestLimiter,
    RobotsCache,
    canonicalize,
    crawl,
    fetch,
    parse_html,
    retry_after_seconds,
)


def test_canonicalize_and_query_order() -> None:
    assert (
        canonicalize("HTTPS://EXAMPLE.org:443/a//b/../?z=2&a=3#frag")
        == "https://example.org/a?a=3&z=2"
    )
    assert (
        canonicalize("/docs/?b=1&a=2", "https://EXAMPLE.org/start")
        == "https://example.org/docs?a=2&b=1"
    )
    assert canonicalize("http://EXAMPLE.ORG:80") == "http://example.org/"
    with pytest.raises(ValueError):
        canonicalize("javascript:alert(1)")


def test_parser_skips_scripts_and_gets_links() -> None:
    text, links = parse_html(
        '<h1>hello</h1><script>secret</script><a href="/other">world</a>'
    )
    assert text == "hello world"
    assert links == ["/other"]


def test_retry_after_and_invalid_configuration() -> None:
    assert retry_after_seconds("2") == 2
    assert retry_after_seconds("not a date") is None
    with pytest.raises(ValueError):
        CrawlConfig("https://example.org", concurrency=0).validate()
    with pytest.raises(ValueError):
        CrawlConfig("https://example.org", retries=4).validate()


@pytest.mark.asyncio
async def test_fetch_429_retries_and_404_does_not() -> None:
    seen = {"retry": 0, "missing": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        route = request.url.path.strip("/")
        seen[route] += 1
        if route == "retry" and seen[route] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if route == "missing":
            return httpx.Response(404)
        return httpx.Response(200, text="good")

    stats = CrawlStats()
    config = CrawlConfig("http://example.test/retry", min_host_delay=0)
    limiter = RequestLimiter(config, stats, lambda: None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        good = await fetch(client, limiter, "http://example.test/retry")
        bad = await fetch(client, limiter, "http://example.test/missing")
    assert (good.status, good.attempts, seen["retry"]) == (200, 2, 2)
    assert (bad.status, bad.attempts, seen["missing"]) == (404, 1, 1)


@pytest.mark.asyncio
async def test_fetch_timeout_retry_bound() -> None:
    calls = 0

    async def slow(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.03)
        return httpx.Response(200)

    stats = CrawlStats()
    config = CrawlConfig("http://example.test/page", timeout_s=0.001, min_host_delay=0)
    limiter = RequestLimiter(config, stats, lambda: None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        result = await fetch(client, limiter, config.seed, retry_limit=1)
    assert result.status == 0 and result.attempts == 2
    assert calls == 2
    assert stats.in_flight == 0


@pytest.mark.asyncio
async def test_robots_cached_and_disallow() -> None:
    robots_hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_hits
        if request.url.path == "/robots.txt":
            robots_hits += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        return httpx.Response(200)

    config = CrawlConfig("http://example.test", min_host_delay=0)
    limiter = RequestLimiter(config, CrawlStats(), lambda: None)
    robots = RobotsCache()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await robots.allowed(
            "http://example.test/public", client, limiter, "FindexLabCrawler", None
        )
        assert not await robots.allowed(
            "http://example.test/private", client, limiter, "FindexLabCrawler", None
        )
    assert robots_hits == 1


@pytest.mark.asyncio
async def test_robots_failure_fails_closed() -> None:
    config = CrawlConfig("http://example.test", min_host_delay=0)
    limiter = RequestLimiter(config, CrawlStats(), lambda: None)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(403))
    ) as client:
        assert not await RobotsCache().allowed(
            config.seed, client, limiter, "FindexLabCrawler", None
        )


@pytest.mark.asyncio
async def test_crawl_mock_deduplicates_and_respects_max_pages() -> None:
    visited: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        visited.append(str(req.url))
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        index = int(req.url.path.rsplit("/", 1)[-1])
        links = "".join(
            f'<a href="/page/{i}?z=1&a=2#test">page{i}</a>'
            for i in range(index + 1, 15)
        )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text=f"<h1>Document {index}</h1>{links}",
        )

    config = CrawlConfig(
        "http://example.test/page/0",
        max_pages=6,
        concurrency=4,
        per_host=2,
        min_host_delay=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        pages = [p async for p in crawl(config, client=client)]
    assert len(pages) == 6
    assert len({p.url for p in pages}) == 6
    assert len([url for url in visited if "robots.txt" in url]) == 1
    assert all(page.status == 200 for page in pages)


@pytest.mark.asyncio
async def test_crawl_denied_never_fetches_page() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, text="User-agent: *\nDisallow: /\n")

    config = CrawlConfig("http://example.test/denied", max_pages=5, min_host_delay=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        pages = [p async for p in crawl(config, client=client)]
    assert pages == []
    assert seen == ["/robots.txt"]


def test_jsonl_document_stream_and_index(tmp_path: Path) -> None:
    source = tmp_path / "site.jsonl"
    source.write_text(
        json.dumps({"url": "http://test/a", "text": "python asyncio python"})
        + "\n"
        + json.dumps({"url": "http://test/b", "text": "asyncio network"})
        + "\n",
        encoding="utf-8",
    )
    docs = list(iter_documents(source))
    assert [doc.doc_id for doc in docs] == ["http://test/a", "http://test/b"]
    index = build_jsonl_index(source)
    assert index.num_docs == 2
    assert index.postings["asyncio"][0].doc_id == 0
    assert index.documents[1].path == "http://test/b"


def test_cli_indexes_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "site.jsonl"
    source.write_text(
        json.dumps({"url": "https://test/page", "text": "python network"}) + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "index.bin"
    result = CliRunner().invoke(app, ["index", str(source), "--out", str(dest)])
    assert result.exit_code == 0, result.output
    assert dest.exists()
    assert "Indexed 1 documents" in result.output
