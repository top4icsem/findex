"""Polite, bounded async HTML crawler; it does not index inside the event loop.

Only visit origins explicitly allowed by the caller. Robots failures fail closed.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
import random
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from time import monotonic, perf_counter
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

LOGGER = logging.getLogger("findex.crawl")
REDIRECTS = {301, 302, 303, 307, 308}
RETRYABLE = {429, 500, 502, 503, 504}


def canonicalize(url: str, base: str | None = None) -> str:
    """Canonicalize HTTP URLs without changing the meaning of query parameters."""
    resolved = urljoin(base, url) if base else url
    parts = urlsplit(resolved)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname or parts.username:
        raise ValueError(f"Not an eligible HTTP URL: {resolved!r}")
    host = parts.hostname.lower()
    if ":" in host:  # IPv6 address
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Invalid port") from exc
    if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    raw_path = re.sub(r"/{2,}", "/", parts.path or "/")
    path = posixpath.normpath(raw_path)
    if not path.startswith("/"):
        path = "/" + path
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(parse_qsl(parts.query, keep_blank_values=True)), doseq=True
    )
    return urlunsplit((scheme, host, path, query, ""))


def origin_of(url: str) -> str:
    pieces = urlsplit(url)
    return f"{pieces.scheme}://{pieces.netloc}"


class PageParser(HTMLParser):
    """HTML standard-library parser, no dependency on native extensions."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.words: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden_depth += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            stripped = " ".join(data.split())
            if stripped:
                self.words.append(stripped)


def parse_html(content: str) -> tuple[str, list[str]]:
    parser = PageParser()
    parser.feed(content)
    return " ".join(parser.words), parser.links


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int
    body: bytes
    elapsed_s: float
    attempts: int
    error: str | None = None
    content_type: str = ""
    location: str = ""


@dataclass(frozen=True, slots=True)
class CrawledPage:
    url: str
    status: int
    text: str
    bytes: int
    elapsed_s: float
    attempts: int

    def as_record(self) -> dict[str, str | int | float]:
        return {
            "url": self.url,
            "status": self.status,
            "text": self.text,
            "bytes": self.bytes,
            "elapsed_s": round(self.elapsed_s, 6),
            "attempts": self.attempts,
        }


@dataclass(slots=True)
class CrawlStats:
    pages: int = 0
    errors: int = 0
    requests: int = 0
    in_flight: int = 0
    queued: int = 0
    started: float = 0.0

    @property
    def pages_per_second(self) -> float:
        return self.pages / max(0.001, perf_counter() - self.started)


@dataclass(frozen=True, slots=True)
class CrawlConfig:
    seed: str
    max_pages: int = 200
    concurrency: int = 5
    per_host: int = 5
    min_host_delay: float = 0.02
    timeout_s: float = 8.0
    retries: int = 3
    user_agent: str = "FindexLabCrawler/0.6 (+local-test)"
    allowed_origins: frozenset[str] = frozenset()

    def validate(self) -> None:
        if self.max_pages < 1 or self.concurrency < 1 or self.per_host < 1:
            raise ValueError("max_pages, concurrency and per_host must be positive")
        if self.timeout_s <= 0 or self.min_host_delay < 0:
            raise ValueError(
                "timeout must be positive; host delay must not be negative"
            )
        if not 0 <= self.retries <= 3:
            raise ValueError("retries must be between 0 and 3")
        if not self.user_agent.strip():
            raise ValueError("A truthful User-Agent is required")
        canonicalize(self.seed)


class RequestLimiter:
    """One global and one per-origin semaphore, with start-to-start spacing."""

    def __init__(
        self, config: CrawlConfig, stats: CrawlStats, notify: Callable[[], None]
    ) -> None:
        self.config = config
        self.stats = stats
        self.notify = notify
        self.global_sem = asyncio.Semaphore(config.concurrency)
        self.host_sems: dict[str, asyncio.Semaphore] = {}
        self.host_locks: dict[str, asyncio.Lock] = {}
        self.next_start: dict[str, float] = {}

    async def get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        host = origin_of(url)
        sem = self.host_sems.setdefault(host, asyncio.Semaphore(self.config.per_host))
        lock = self.host_locks.setdefault(host, asyncio.Lock())
        async with self.global_sem, sem:
            async with lock:
                wait = self.next_start.get(host, 0.0) - monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                self.next_start[host] = monotonic() + self.config.min_host_delay
            self.stats.in_flight += 1
            self.stats.requests += 1
            self.notify()
            try:
                async with asyncio.timeout(self.config.timeout_s):
                    return await client.get(url, follow_redirects=False)
            finally:
                self.stats.in_flight -= 1
                self.notify()


def retry_after_seconds(header: str | None) -> float | None:
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        try:
            instant = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            return None
        if instant.tzinfo is None:
            instant = instant.replace(tzinfo=UTC)
        return max(0.0, (instant - datetime.now(UTC)).total_seconds())


async def fetch(
    client: httpx.AsyncClient,
    limiter: RequestLimiter,
    url: str,
    *,
    retry_limit: int = 3,
    on_attempt: Callable[[str, int, int, float, int, str | None], None] | None = None,
) -> FetchResult:
    """Retry only 429, 5xx and timeouts; allow up to retry_limit *retries*."""
    first = perf_counter()
    for attempt in range(1, retry_limit + 2):
        start = perf_counter()
        status = 0
        body = b""
        error: str | None = None
        retry_after: float | None = None
        location = ""
        content_type = ""
        try:
            response = await limiter.get(client, url)
            status = response.status_code
            body = response.content
            content_type = response.headers.get("content-type", "")
            location = response.headers.get("location", "")
            retry_after = retry_after_seconds(response.headers.get("retry-after"))
        except (httpx.TimeoutException, TimeoutError) as exc:
            error = f"timeout: {type(exc).__name__}"
        except httpx.RequestError as exc:
            error = f"request: {type(exc).__name__}: {exc}"
        elapsed = perf_counter() - start
        if on_attempt:
            on_attempt(url, status, len(body), elapsed, attempt, error)
        retryable = (
            status == 429
            or status >= 500
            or error is not None
            and error.startswith("timeout:")
        )
        if not retryable or attempt > retry_limit:
            return FetchResult(
                url,
                status,
                body,
                perf_counter() - first,
                attempt,
                error,
                content_type,
                location,
            )
        delay = min(8.0, 0.2 * 2 ** (attempt - 1)) + random.uniform(0, 0.05)
        if retry_after is not None:
            delay = max(delay, retry_after)
        LOGGER.info(
            "RETRY url=%s status=%s attempt=%s delay_s=%.3f",
            url,
            status,
            attempt,
            delay,
        )
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")


class RobotsCache:
    """Per-origin robots cache, with serialized first fetch and fail-closed errors."""

    def __init__(self) -> None:
        self._rules: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(
        self,
        url: str,
        client: httpx.AsyncClient,
        limiter: RequestLimiter,
        user_agent: str,
        on_attempt: Callable[[str, int, int, float, int, str | None], None] | None,
    ) -> bool:
        origin = origin_of(url)
        if origin not in self._rules:
            async with self._locks.setdefault(origin, asyncio.Lock()):
                if origin not in self._rules:
                    robot_url = origin + "/robots.txt"
                    robots = await fetch(
                        client, limiter, robot_url, on_attempt=on_attempt
                    )
                    if robots.status == 404:
                        rule = RobotFileParser()
                        rule.parse([])  # missing robots means allowed
                        self._rules[origin] = rule
                    elif robots.status == 200:
                        rule = RobotFileParser()
                        rule.parse(
                            robots.body.decode("utf-8", errors="replace").splitlines()
                        )
                        self._rules[origin] = rule
                    else:
                        self._rules[origin] = None  # fail closed
                        LOGGER.warning(
                            "ROBOTS_UNAVAILABLE origin=%s status=%s",
                            origin,
                            robots.status,
                        )
        rules = self._rules[origin]
        return rules is not None and rules.can_fetch(user_agent, url)


async def crawl(
    config: CrawlConfig,
    *,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    on_progress: Callable[[CrawlStats], None] | None = None,
    on_attempt: Callable[[str, int, int, float, int, str | None], None] | None = None,
) -> AsyncIterator[CrawledPage]:
    """Yield pages incrementally; workers live in a TaskGroup and are cancelled on exit."""
    config.validate()
    seed = canonicalize(config.seed)
    origins = config.allowed_origins or frozenset({origin_of(seed)})
    if origin_of(seed) not in origins:
        raise ValueError("Seed origin must be allowed")
    current_stats = stats if stats is not None else CrawlStats()
    current_stats.started = perf_counter()
    queue: asyncio.Queue[str] = asyncio.Queue()
    output: asyncio.Queue[CrawledPage | None] = asyncio.Queue(
        maxsize=max(2, config.concurrency * 2)
    )
    seen: set[str] = {seed}
    queue.put_nowait(seed)
    rules = RobotsCache()

    def update() -> None:
        current_stats.queued = queue.qsize()
        if on_progress:
            on_progress(current_stats)

    limiter = RequestLimiter(config, current_stats, update)

    def permitted(candidate: str) -> bool:
        return origin_of(candidate) in origins

    async def worker(http: httpx.AsyncClient) -> None:
        while True:
            url = await queue.get()
            try:
                if not await rules.allowed(
                    url, http, limiter, config.user_agent, on_attempt
                ):
                    LOGGER.warning("ROBOTS_DENIED url=%s", url)
                    current_stats.errors += 1
                    continue
                response = await fetch(
                    http,
                    limiter,
                    url,
                    retry_limit=config.retries,
                    on_attempt=on_attempt,
                )
                if response.status in REDIRECTS and response.location:
                    try:
                        redirected = canonicalize(response.location, url)
                    except ValueError:
                        current_stats.errors += 1
                        LOGGER.warning("BAD_REDIRECT url=%s", url)
                    else:
                        if (
                            permitted(redirected)
                            and redirected not in seen
                            and len(seen) < config.max_pages
                        ):
                            seen.add(redirected)
                            queue.put_nowait(redirected)
                        else:
                            LOGGER.info("SKIP_REDIRECT url=%s dest=%s", url, redirected)
                    continue
                if response.status != 200 or response.error:
                    current_stats.errors += 1
                    LOGGER.warning(
                        "FETCH_FAILED url=%s status=%s error=%s",
                        url,
                        response.status,
                        response.error,
                    )
                    continue
                if "text/html" not in response.content_type and response.content_type:
                    LOGGER.info("SKIP_NONHTML url=%s", url)
                    continue
                text, links = parse_html(
                    response.body.decode("utf-8", errors="replace")
                )
                for href in links:
                    if len(seen) >= config.max_pages:
                        break
                    try:
                        child = canonicalize(href, url)
                    except ValueError:
                        continue
                    if permitted(child) and child not in seen:
                        seen.add(child)  # no await between membership test and add
                        queue.put_nowait(child)
                await output.put(
                    CrawledPage(
                        url,
                        200,
                        text,
                        len(response.body),
                        response.elapsed_s,
                        response.attempts,
                    )
                )
            finally:
                queue.task_done()
                update()

    async def close_when_done() -> None:
        await queue.join()
        await output.put(None)

    owns_client = client is None
    http = (
        client
        if client is not None
        else httpx.AsyncClient(
            headers={"User-Agent": config.user_agent},
            timeout=None,  # outer asyncio.timeout handles timing
            max_redirects=5,
        )
    )
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(worker(http)) for _ in range(config.concurrency)]
            closer = group.create_task(close_when_done())
            try:
                while True:
                    item = await output.get()
                    if item is None:
                        break
                    current_stats.pages += 1
                    update()
                    yield item
                    if current_stats.pages >= config.max_pages:
                        break
            finally:
                for task in tasks:
                    task.cancel()
                closer.cancel()
    finally:
        if owns_client:
            await http.aclose()
