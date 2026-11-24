"""Convert actual oha JSON output into the required README table."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("docs/lab07_oha")
CASES = [
    ("1 worker / async def", "one_async.json"),
    ("1 worker / def", "one_sync.json"),
    ("4 workers / def", "four_workers.json"),
    ("Public deployment", "deployed.json"),
]
print("| Configuration | Requests/s | p50 (ms) | p95 (ms) | p99 (ms) |")
print("|---|---:|---:|---:|---:|")
for label, filename in CASES:
    path = ROOT / filename
    if not path.exists():
        print(f"| {label} | — | — | — | — |")
        continue
    content = json.loads(path.read_text(encoding="utf-8"))
    metrics = content.get("metrics", {})
    latencies = metrics.get("latency_ms", {})
    if not latencies:
        latencies = {
            k: value * 1000 for k, value in content["latencyPercentiles"].items()
        }
    rate = metrics.get("requests_per_sec", content["summary"]["requestsPerSec"])
    print(
        f"| {label} | {rate:.2f} | {latencies['p50']:.2f} | "
        f"{latencies['p95']:.2f} | {latencies['p99']:.2f} |"
    )
