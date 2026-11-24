"""Owned loopback-only 240-page site for reproducible Lab 06 measurements."""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import urlsplit


class LabHandler(BaseHTTPRequestHandler):
    pages = 240
    delay_s = 0.03
    hits: ClassVar[dict[str, int]] = {}
    hit_lock = threading.Lock()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/robots.txt":
            self.respond(
                200, "text/plain", b"User-agent: *\nAllow: /\nDisallow: /blocked\n"
            )
            return
        if path == "/contact":
            self.respond(
                200,
                "text/html",
                b"<html><body>Local site owner contact endpoint</body></html>",
            )
            return
        if path == "/blocked":
            self.respond(
                200, "text/html", b"<html><body>Not to be crawled</body></html>"
            )
            return
        if not path.startswith("/page/"):
            self.respond(404, "text/plain", b"Not found")
            return
        try:
            number = int(path.rsplit("/", 1)[1])
        except ValueError:
            self.respond(404, "text/plain", b"Invalid page")
            return
        if number < 0 or number >= self.pages:
            self.respond(404, "text/plain", b"Not found")
            return
        with self.hit_lock:
            count = self.hits.get(path, 0) + 1
            self.hits[path] = count
        if number == 5 and count == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        time.sleep(self.delay_s)
        links = "".join(
            f'<a href="/page/{child:03d}?b=2&a=1#section">Document {child}</a>'
            for child in range(number + 1, min(self.pages, number + 21))
        )
        page = (
            f"<html><head><title>Index page {number}</title></head>"
            f"<body><h1>Findex local async crawler document {number}</h1>"
            f"<p>Asyncio exercises network waiting and HTTP request concurrency.</p>"
            f"{links}</body></html>"
        ).encode()
        self.respond(200, "text/html; charset=utf-8", page)

    def respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def start_local_site(
    delay_s: float = 0.03,
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """Bind only 127.0.0.1, never expose the demo website externally."""
    handler = type(
        "LocalLabHandler",
        (LabHandler,),
        {
            "delay_s": delay_s,
            "hits": {},
            "hit_lock": threading.Lock(),
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local authorized 240-page demo site")
    parser.add_argument("--delay", type=float, default=0.03)
    args = parser.parse_args()
    server, thread, origin = start_local_site(args.delay)
    print(f"Local authorized site: {origin}/page/000", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
