"""Optional MiniLM chunk embeddings, cosine retrieval, and RRF hybrid search.

The model is deliberately optional: regular Findex and Render Free can start
without torch. Enable locally with `uv sync --extra semantic` and run `findex embed`.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from findex.index import Index
from findex.numpy_rank import top_indices
from findex.ranking import SearchResult, make_snippet
from findex.tokenize import tokenize

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
Float32Array = NDArray[np.float32]


class Encoder(Protocol):
    def encode(self, sentences: str | list[str], **kwargs: Any) -> Any: ...


def load_encoder(model_name: str = MODEL) -> Encoder:
    """Import the optional extra at first semantic use, never on API startup."""
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional model dependencies: uv sync --extra semantic"
        ) from exc
    return cast(Encoder, module.SentenceTransformer(model_name, device="cpu"))


def _unit_rows(matrix: Float32Array) -> Float32Array:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / np.maximum(norms, 1e-12), dtype=np.float32)


def _chunk_words(text: str, size: int = 160, overlap: int = 32) -> list[str]:
    """Whitespace chunks preserve the text used by the embedding model."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Expect chunk_size > 0 and 0 <= overlap < chunk_size")
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    for start in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def _text_for(index: Index, doc_id: int) -> str:
    meta = index.documents[doc_id]
    if index.corpus_root:
        root = Path(index.corpus_root).resolve()
        path = (root / meta.path).resolve()
        if path.is_relative_to(root) and path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    # For JSONL crawls, local source may no longer be accessible.
    return meta.preview


def make_embeddings(
    index: Index,
    output: Path,
    *,
    model_name: str = MODEL,
    chunk_size: int = 160,
    overlap: int = 32,
    batch_size: int = 32,
    encoder: Encoder | None = None,
) -> tuple[int, int]:
    """Write normalized [chunks, dims] .npy and doc-ID sidecar JSON.

    No untrusted pickle or JSON-supplied executable code is loaded.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chunks: list[str] = []
    owners: list[int] = []
    for doc_id in sorted(index.documents):
        parts = _chunk_words(_text_for(index, doc_id), chunk_size, overlap)
        for part in parts:
            chunks.append(part)
            owners.append(doc_id)
    if not chunks:
        raise ValueError("Cannot embed an empty text collection")
    model = encoder if encoder is not None else load_encoder(model_name)
    matrices: list[Float32Array] = []
    for start in range(0, len(chunks), batch_size):
        matrix = np.asarray(
            model.encode(chunks[start : start + batch_size], convert_to_numpy=True),
            dtype=np.float32,
        )
        if matrix.ndim != 2:
            raise ValueError("Model must return a [batch, dimension] array")
        matrices.append(matrix)
    vectors = _unit_rows(np.concatenate(matrices, axis=0))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".npy":
        raise ValueError("Embedding matrix filename must end with .npy")
    np.save(output, vectors, allow_pickle=False)
    sidecar = {
        "version": 1,
        "model": model_name,
        "doc_ids": owners,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "dim": int(vectors.shape[1]),
        "num_docs": index.num_docs,
    }
    output.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
    )
    return len(owners), vectors.shape[1]


@dataclass(slots=True)
class EmbeddingStore:
    vectors: Float32Array
    chunk_doc_ids: NDArray[np.int32]
    model_name: str
    encoder: Encoder | None = None

    @classmethod
    def load(cls, source: Path, index: Index | None = None) -> EmbeddingStore:
        meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("version") != 1:
            raise ValueError("Unsupported embedding metadata version")
        matrix = np.load(source, allow_pickle=False, mmap_mode="r")
        ids = np.asarray(meta["doc_ids"], dtype=np.int32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("Embedding matrix and doc-ID metadata disagree")
        if int(meta["dim"]) != matrix.shape[1]:
            raise ValueError("Embedding dimensions do not match metadata")
        if index is not None:
            if len(index.documents) != int(meta["num_docs"]):
                raise ValueError("Embedding data belongs to a different corpus")
            if not set(map(int, ids)).issubset(index.documents):
                raise ValueError("Embedding data has unknown document IDs")
        return cls(cast(Float32Array, matrix), ids, str(meta["model"]))

    def rank_ids(self, query: str, k: int) -> list[tuple[int, float]]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        if k == 0 or not query.strip():
            return []
        if self.encoder is None:
            self.encoder = load_encoder(self.model_name)
        raw = np.asarray(
            self.encoder.encode([query], convert_to_numpy=True), dtype=np.float32
        )
        if raw.ndim != 2 or raw.shape[0] != 1 or raw.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query embedding dimension mismatch")
        query_vec = _unit_rows(raw)[0]
        cosine = self.vectors @ query_vec
        doc_ids, reverse = np.unique(self.chunk_doc_ids, return_inverse=True)
        best = np.full(len(doc_ids), -np.inf, dtype=np.float32)
        np.maximum.at(best, reverse, cosine)
        selected = top_indices(doc_ids, best.astype(np.float64), k)
        return [(int(doc_ids[i]), float(best[i])) for i in selected]


def reciprocal_rank_fusion(
    lexical: list[tuple[int, float]],
    semantic: list[tuple[int, float]],
    *,
    constant: int = 60,
) -> list[tuple[int, float]]:
    """1/(k+rank) from both ordered rankings; deterministic doc-ID ties."""
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    points: dict[int, float] = {}
    for ranking in (lexical, semantic):
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            points[doc_id] = points.get(doc_id, 0.0) + 1.0 / (constant + rank)
    return sorted(points.items(), key=lambda row: (-row[1], row[0]))


def semantic_results(
    index: Index, store: EmbeddingStore, query: str, top_k: int
) -> list[SearchResult]:
    words = set(tokenize(query))
    return [
        SearchResult(
            doc_id=doc_id,
            score=score,
            path=index.documents[doc_id].path,
            snippet=make_snippet(index, doc_id, words),
        )
        for doc_id, score in store.rank_ids(query, top_k)
    ]
