import sys
from pathlib import Path

from findex.index import build_index
from findex.open_index import open_index
from findex.ranking import RankedSearcher
from findex.scoring import BM25

if sys.argv[1] == "build":
    for _ in range(4):
        build_index(Path("data"))
elif sys.argv[1] == "search":
    with open_index(Path("index.bin")) as index:
        searcher = RankedSearcher(index, BM25())
        for _ in range(2000):
            searcher.search(
                "python AND search AND engine",
                top_k=5,
            )
