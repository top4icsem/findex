"""Controlled lost-update demonstration; works with or without the GIL."""

from __future__ import annotations

import sys
import threading
from collections import Counter


def demonstrate(iterations: int = 1000) -> tuple[int, int, int]:
    shared: Counter[str] = Counter()
    gate = threading.Barrier(2)

    def unsafe() -> None:
        for _ in range(iterations):
            current = shared["hits"]
            gate.wait()  # Force both threads to observe the same old value.
            shared["hits"] = current + 1
            gate.wait()

    threads = [threading.Thread(target=unsafe) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    lost_result = shared["hits"]

    shared.clear()
    lock = threading.Lock()

    def safe() -> None:
        for _ in range(iterations):
            with lock:
                shared["hits"] += 1

    threads = [threading.Thread(target=safe) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return iterations * 2, lost_result, shared["hits"]


def main() -> None:
    checker = getattr(sys, "_is_gil_enabled", None)
    print(f"GIL enabled: {checker() if callable(checker) else 'not supported'}")
    expected, unsafe, safe = demonstrate()
    print(
        f"Expected: {expected}; unsynchronized (forced interleaving): {unsafe}; locked: {safe}"
    )
    print(
        "The forced interleaving demonstrates that GIL does not make read-modify-write atomic."
    )


if __name__ == "__main__":
    main()
