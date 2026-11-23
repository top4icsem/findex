"""Interchangeable TF-IDF and BM25 scorers via structural typing."""

from dataclasses import dataclass
from math import log, log1p, sqrt
from typing import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    def score(
        self,
        *,
        tf: int,
        df: int,
        doc_length: int,
        avg_doc_length: float,
        num_docs: int,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class Tfidf:
    """Logarithmic TF, smoothed IDF, and sqrt(length) normalization."""

    def score(
        self,
        *,
        tf: int,
        df: int,
        doc_length: int,
        avg_doc_length: float,
        num_docs: int,
    ) -> float:
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        idf = log((num_docs + 1) / (df + 1)) + 1.0
        return (1.0 + log(tf)) * idf / sqrt(max(1, doc_length))


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(
        self,
        *,
        tf: int,
        df: int,
        doc_length: int,
        avg_doc_length: float,
        num_docs: int,
    ) -> float:
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        idf = log1p((num_docs - df + 0.5) / (df + 0.5))
        avg = max(avg_doc_length, 1.0)
        saturation = self.k1 * (1.0 - self.b + self.b * doc_length / avg)
        return idf * (tf * (self.k1 + 1.0)) / (tf + saturation)
