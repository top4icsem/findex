"""cProfile baseline+post-change evidence for build and search.

The legacy Lab 03 ranker is intentionally retained and is profiled separately.
Do not replace recorded measurements by example numbers.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
from pathlib import Path

from findex.index import build_index
from findex.numpy_rank import NumpyRanker
from findex.open_index import open_index
from findex.ranking import RankedSearcher
from findex.scoring import BM25


def capture(path: Path, target: object) -> None:
    profile = cProfile.Profile()
    profile.enable()
    target()  # type: ignore[operator]
    profile.disable()
    profile.dump_stats(str(path.with_suffix(".pstats")))
    with path.with_suffix(".txt").open("w", encoding="utf-8") as stream:
        from contextlib import redirect_stdout

        with redirect_stdout(stream):
            pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(30)
    print("Saved:", path.with_suffix(".pstats"), path.with_suffix(".txt"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data"))
    ap.add_argument("--index", type=Path, default=Path("index.bin"))
    ap.add_argument("--query", type=str, default="python AND search AND engine")
    ap.add_argument("--out", type=Path, default=Path("docs/lab08_profile"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    capture(args.out / "build", lambda: build_index(args.corpus))
    with open_index(args.index) as index:
        capture(
            args.out / "legacy_search",
            lambda: RankedSearcher(index, BM25()).search(args.query, top_k=5),
        )
        capture(
            args.out / "numpy_search",
            lambda: NumpyRanker(index, BM25()).search(args.query, top_k=5),
        )


if __name__ == "__main__":
    main()
