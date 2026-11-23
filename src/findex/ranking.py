"""Rank Boolean matches with interchangeable scorers and cached query evaluation."""

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from heapq import nlargest
from pathlib import Path

from findex.index import Index
from findex.open_index import open_index
from findex.query_ast import parse_query, positive_terms
from findex.scoring import BM25, Scorer, Tfidf
from findex.timing import timed

logger = logging.getLogger("findex.cache")


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    """Order by high score, then low doc_id for deterministic ties."""

    sort_key: tuple[float, int] = field(init=False, repr=False)
    doc_id: int = field(compare=False)
    score: float = field(compare=False)
    path: str = field(compare=False)
    snippet: str = field(compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_key", (self.score, -self.doc_id))


def make_snippet(index: Index, doc_id: int, terms: set[str], window: int = 80) -> str:
    """Read the source document and highlight terms near the rarest match."""
    meta = index.documents[doc_id]
    path = Path(index.corpus_root) / meta.path if index.corpus_root else None
    try:
        text = path.read_text(encoding="utf-8") if path else meta.preview
    except OSError:
        text = meta.preview  # Offline fallback, limited to the original preview.

    if not text:
        return ""

    patterns = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
    if not patterns:
        return " ".join(text[: 2 * window].split())
    matches = re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.IGNORECASE)
    found = list(matches.finditer(text))
    if found:
        # Prefer a rare term, then the first position in the document.
        best = min(found, key=lambda m: (index.df(m.group().casefold()), m.start()))
        start = max(0, best.start() - window)
        stop = min(len(text), best.end() + window)
    else:
        start, stop = 0, min(len(text), 2 * window)

    fragment = text[start:stop]
    fragment = matches.sub(lambda m: f"**{m.group(0)}**", fragment)
    fragment = " ".join(fragment.split())
    return ("…" if start else "") + fragment + ("…" if stop < len(text) else "")


class RankedSearcher:
    def __init__(self, index: Index, scorer: Scorer | None = None):
        self.index = index
        self.scorer = scorer if scorer is not None else BM25()

        @lru_cache(maxsize=256)
        def matching_ids(query: str) -> tuple[int, ...]:
            return tuple(sorted(parse_query(query).evaluate(index)))

        self._matching_ids = matching_ids

    @timed
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        hits_before = self._matching_ids.cache_info().hits
        doc_ids = self._matching_ids(query)
        cache_hit = self._matching_ids.cache_info().hits > hits_before
        logger.info("CACHE %s: %r", "HIT" if cache_hit else "MISS", query)

        terms = positive_terms(parse_query(query))
        frequencies = {
            term: {posting.doc_id: posting.frequency for posting in self.index[term]}
            for term in terms
            if term in self.index
        }

        # Compute numeric scores only for the Boolean-matching documents.
        scored = []
        for doc_id in doc_ids:
            meta = self.index.documents[doc_id]
            score = sum(
                self.scorer.score(
                    tf=by_doc.get(doc_id, 0),
                    df=self.index.df(term),
                    doc_length=meta.length,
                    avg_doc_length=self.index.avg_doc_length,
                    num_docs=self.index.num_docs,
                )
                for term, by_doc in frequencies.items()
            )
            scored.append((score, -doc_id, doc_id))

        # Avoid generating expensive snippets for documents outside the top-k.
        winners = nlargest(top_k, scored)
        return [
            SearchResult(
                doc_id=doc_id,
                score=score,
                path=self.index.documents[doc_id].path,
                snippet=make_snippet(self.index, doc_id, terms),
            )
            for score, _, doc_id in winners
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank documents using TF-IDF or BM25.")
    parser.add_argument("index_file", type=Path)
    parser.add_argument("query")
    parser.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat to demonstrate cache hits"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print @timed and cache logs"
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.top < 0:
        parser.error("--repeat must be positive and --top nonnegative")
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    scorer = BM25() if args.scorer == "bm25" else Tfidf()
    with open_index(args.index_file) as index:
        ranked = RankedSearcher(index, scorer)
        for number in range(args.repeat):
            try:
                results = ranked.search(args.query, top_k=args.top)
            except ValueError as exc:
                parser.error(str(exc))
            print(f"Run: {number + 1}; scorer: {args.scorer}; results: {len(results)}")
            for result in results:
                print(f"  doc={result.doc_id} score={result.score:.6f} {result.path}")
                if sys.stdout.isatty():
                    display = re.sub(
                        r"\*\*(.*?)\*\*",
                        lambda match: f"\033[1;33m{match.group(1)}\033[0m",
                        result.snippet,
                    )
                else:
                    display = result.snippet
                print(f"    {display}")
        print(f"Cache stats: {ranked._matching_ids.cache_info()}")


if __name__ == "__main__":
    main()
