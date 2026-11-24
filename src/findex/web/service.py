"""Lab 08 search service: fast lexical ranking plus optional semantic/RRF modes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from findex.index import Index
from findex.numpy_rank import NumpyRanker
from findex.query_ast import parse_query
from findex.ranking import SearchResult, make_snippet
from findex.scoring import BM25, Tfidf
from findex.semantic import EmbeddingStore, reciprocal_rank_fusion, semantic_results
from findex.tokenize import tokenize
from findex.web.models import DocumentModel, ResultModel, SearchPage

Mode = Literal["lexical", "semantic", "hybrid"]
ScoreName = Literal["bm25", "tfidf"]


class SearchService:
    def __init__(self, index: Index, embeddings: EmbeddingStore | None = None):
        self.index = index
        self.embeddings = embeddings
        self.searchers = {
            "bm25": NumpyRanker(index, BM25()),
            "tfidf": NumpyRanker(index, Tfidf()),
        }

    def ranked(
        self, q: str, top_k: int, scorer: ScoreName, mode: Mode
    ) -> list[SearchResult]:
        if mode == "lexical":
            return self.searchers[scorer].search(q, top_k=top_k)
        if self.embeddings is None:
            raise ValueError(
                "Semantic index not configured: run 'findex embed' and set EMBEDDINGS_PATH"
            )
        if mode == "semantic":
            return semantic_results(self.index, self.embeddings, q, top_k)
        # Natural-language queries aren't necessarily Boolean expressions.
        try:
            parse_query(q)
            lex_query = q
        except ValueError:
            lex_query = " OR ".join(dict.fromkeys(tokenize(q)))
        lexical = (
            self.searchers[scorer].rank_ids(lex_query, self.index.num_docs)
            if lex_query
            else []
        )
        dense = self.embeddings.rank_ids(q, self.index.num_docs)
        merged = reciprocal_rank_fusion(lexical, dense)[:top_k]
        words = set(tokenize(q))
        return [
            SearchResult(
                doc_id=doc_id,
                score=score,
                path=self.index.documents[doc_id].path,
                snippet=make_snippet(self.index, doc_id, words),
            )
            for doc_id, score in merged
        ]

    def search(
        self, q: str, k: int, scorer: ScoreName, page: int, mode: Mode = "lexical"
    ) -> SearchPage:
        if mode == "lexical":
            total = len(parse_query(q).evaluate(self.index))
        else:
            if self.embeddings is None:
                raise ValueError("Semantic index not configured; set EMBEDDINGS_PATH")
            total = len(set(map(int, self.embeddings.chunk_doc_ids)))
            if mode == "hybrid":
                total = self.index.num_docs
        hits = self.ranked(q, min(total, k * page), scorer, mode)
        items = hits[(page - 1) * k : page * k]
        return SearchPage(
            q=q,
            k=k,
            scorer=scorer,
            mode=mode,
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
                for item in items
            ],
        )

    def document(self, doc_id: int) -> DocumentModel | None:
        meta = self.index.documents.get(doc_id)
        if meta is None:
            return None
        text = meta.preview
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
