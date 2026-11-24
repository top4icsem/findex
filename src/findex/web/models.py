"""Validation and serialization at the HTTP boundary only."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    q: str = Field(min_length=1, max_length=500)
    k: int = Field(default=10, ge=1, le=50)
    scorer: Literal["bm25", "tfidf"] = "bm25"
    page: int = Field(default=1, ge=1, le=1000)


class ResultModel(BaseModel):
    doc_id: int
    path: str
    score: float
    snippet: str


class SearchPage(BaseModel):
    q: str
    scorer: Literal["bm25", "tfidf"]
    k: int
    page: int
    total: int
    pages: int
    results: list[ResultModel]


class DocumentModel(BaseModel):
    doc_id: int
    path: str
    length: int
    text: str


class StatsModel(BaseModel):
    documents: int
    terms: int
    avg_doc_length: float
    positions: bool


class HealthModel(BaseModel):
    status: Literal["ok"]
    documents: int
