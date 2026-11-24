"""FastAPI app with lifespan index, validated API and accessible search UI."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from findex.semantic import EmbeddingStore
from findex.storage import load_index
from findex.web.models import (
    DocumentModel,
    HealthModel,
    SearchPage,
    SearchQuery,
    StatsModel,
)
from findex.web.service import SearchService
from findex.web.settings import Settings

logger = logging.getLogger("findex.web")
WEB_ROOT = Path(__file__).parent


def get_service(request: Request) -> SearchService:
    service: SearchService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return service


Service = Annotated[SearchService, Depends(get_service)]


def get_search_query(
    q: Annotated[str, Query(min_length=1, max_length=500)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    scorer: Literal["bm25", "tfidf"] = "bm25",
    mode: Literal["lexical", "semantic", "hybrid"] = "lexical",
    page: Annotated[int, Query(ge=1, le=1000)] = 1,
) -> SearchQuery:
    return SearchQuery(q=q, k=k, scorer=scorer, mode=mode, page=page)


SearchParams = Annotated[SearchQuery, Depends(get_search_query)]


def highlight(value: str) -> Markup:
    """Escape all source content before interpreting our **term** markers."""
    segments = re.split(r"(\*\*.*?\*\*)", value)
    rendered: list[str] = []
    for segment in segments:
        if segment.startswith("**") and segment.endswith("**"):
            rendered.append(f"<mark>{escape(segment[2:-2])}</mark>")
        else:
            rendered.append(str(escape(segment)))
    return Markup("".join(rendered))


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        application.state.service = None
        if config.index_path.is_file():
            # This runs once per worker on startup; never per HTTP request.
            try:
                file_format = (
                    "json" if config.index_path.suffix.lower() == ".json" else "pickle"
                )
                index = load_index(config.index_path, file_format)
                embeddings = None
                if (
                    config.embeddings_path is not None
                    and config.embeddings_path.is_file()
                ):
                    embeddings = EmbeddingStore.load(config.embeddings_path, index)
                application.state.service = SearchService(index, embeddings)
                logger.info(
                    "Loaded %d documents from %s", index.num_docs, config.index_path
                )
            except (OSError, ValueError, UnicodeError):
                logger.exception("Could not load index from %s", config.index_path)
        else:
            logger.error("Index file not found: %s", config.index_path)
        try:
            yield
        finally:
            service: SearchService | None = application.state.service
            if service is not None:
                service.index.close()
            application.state.service = None

    application = FastAPI(
        title="Findex Search API",
        version="1.0.0",
        description="Fast NumPy lexical ranking, MiniLM semantic search, and RRF hybrid.",
        lifespan=lifespan,
    )
    application.state.service = None
    templates = Jinja2Templates(directory=str(WEB_ROOT / "templates"))
    templates.env.filters["highlight"] = highlight
    application.mount(
        "/static", StaticFiles(directory=WEB_ROOT / "static"), name="static"
    )

    @application.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Always generate the ID server-side, not from untrusted headers.
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.exception_handler(Exception)
    async def unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error("Unhandled error request_id=%s", request_id, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    @application.get("/health", response_model=HealthModel)
    def health(request: Request) -> HealthModel:
        service: SearchService | None = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Search index is not loaded")
        return HealthModel(status="ok", documents=service.index.num_docs)

    @application.get("/stats", response_model=StatsModel)
    def stats(service: Service) -> StatsModel:
        index = service.index
        return StatsModel(
            documents=index.num_docs,
            terms=len(index),
            avg_doc_length=index.avg_doc_length,
            positions=index.with_positions,
        )

    @application.get("/docs/{doc_id}", response_model=DocumentModel)
    def document(doc_id: int, service: Service) -> DocumentModel:
        result = service.document(doc_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return result

    @application.get("/search", response_model=SearchPage)
    def search(params: SearchParams, service: Service) -> SearchPage:
        # Normal def: FastAPI offloads CPU-bound ranking to the threadpool.
        try:
            return service.search(
                params.q, params.k, params.scorer, params.page, params.mode
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.get("/", response_class=HTMLResponse)
    def home(
        request: Request,
        service: Service,
        q: Annotated[str, Query(max_length=500)] = "",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        mode: Literal["lexical", "semantic", "hybrid"] = "lexical",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> Response:
        results: SearchPage | None = None
        error: str | None = None
        if q.strip():
            try:
                results = service.search(q.strip(), k, scorer, page, mode)
            except ValueError as exc:
                error = str(exc)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "q": q,
                "k": k,
                "page": page,
                "scorer": scorer,
                "mode": mode,
                "results": results,
                "error": error,
                "num_docs": service.index.num_docs,
            },
        )

    @application.get("/view/{doc_id}", response_class=HTMLResponse)
    def view_document(request: Request, doc_id: int, service: Service) -> Response:
        result = service.document(doc_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return templates.TemplateResponse(
            request=request,
            name="document.html",
            context={"document": result},
        )

    if config.enable_bench:

        @application.get(
            "/_bench/sync", response_model=SearchPage, include_in_schema=False
        )
        def bench_sync(params: SearchParams, service: Service) -> SearchPage:
            return service.search(
                params.q, params.k, params.scorer, params.page, params.mode
            )

        @application.get(
            "/_bench/async", response_model=SearchPage, include_in_schema=False
        )
        async def bench_async(params: SearchParams, service: Service) -> SearchPage:
            # INTENTIONALLY blocking: only for comparing the scheduling effect.
            return service.search(
                params.q, params.k, params.scorer, params.page, params.mode
            )

    return application


app = create_app()
