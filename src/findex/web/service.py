"""Search logic; keep internal records as dataclasses from previous labs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from findex.index import Index
from findex.query_ast import parse_query
from findex.ranking import RankedSearcher
from findex.scoring import BM25, Tfidf
from findex.web.models import DocumentModel, ResultModel, SearchPage


class SearchService:
    def __init__(self, index: Index):
        self.index = index
        self.searchers = {
            "bm25": RankedSearcher(index, BM25()),
            "tfidf": RankedSearcher(index, Tfidf()),
        }

    def search(
        self, q: str, k: int, scorer: Literal["bm25", "tfidf"], page: int
    ) -> SearchPage:
        # Parse the Boolean query once for exact matches. Ranking stays in Lab 03.
        total = len(parse_query(q).evaluate(self.index))
        results = self.searchers[scorer].search(q, top_k=min(total, k * page))
        chunk = results[(page - 1) * k : page * k]
        return SearchPage(
            q=q,
            k=k,
            scorer=scorer,
            page=page,
            total=total,
            pages=(total + k - 1) // k,
            results=[
                ResultModel(
                    doc_id=item.doc_id,
                    path=item.path,
                    score=item.score,
                    snippet=item.snippet,
                )
                for item in chunk
            ],
        )

    def document(self, doc_id: int) -> DocumentModel | None:
        meta = self.index.documents.get(doc_id)
        if meta is None:
            return None
        text = meta.preview
        # Never trust an absolute or parent-traversing path from saved index data.
        root = Path(self.index.corpus_root) if self.index.corpus_root else None
        relative = Path(meta.path)
        if root is not None and root.is_dir() and not relative.is_absolute():
            try:
                target = (root / relative).resolve()
                if target.is_relative_to(root.resolve()) and target.is_file():
                    text = target.read_text(encoding="utf-8")
            except (OSError, UnicodeError, ValueError):
                pass
        return DocumentModel(
            doc_id=doc_id, path=meta.path, length=meta.length, text=text
        )
