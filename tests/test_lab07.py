"""Lab 07 HTTP contract tests; all in-memory, no real index file or network."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import DocMeta, Index, Posting
from findex.storage import save_index
from findex.web.app import create_app, get_service
from findex.web.service import SearchService
from findex.web.settings import Settings


@pytest.fixture
def service() -> SearchService:
    index = Index(
        postings={
            "python": [Posting(0, 2), Posting(1, 1)],
            "search": [Posting(0, 1)],
            "asyncio": [Posting(1, 2)],
        },
        documents={
            0: DocMeta(0, "first.txt", 6, "Python search Python application."),
            1: DocMeta(1, "second.txt", 6, "Python asyncio example."),
        },
    )
    return SearchService(index)


@pytest.fixture
def client(service: SearchService) -> TestClient:
    application = create_app(Settings(index_path=Path("/absent/findex-lab07.json")))
    application.dependency_overrides[get_service] = lambda: service
    with TestClient(application) as http:
        yield http
    application.dependency_overrides.clear()


def test_search_success(client: TestClient) -> None:
    response = client.get("/search", params={"q": "python", "k": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["pages"] == 2
    assert len(data["results"]) == 1
    assert data["results"][0]["doc_id"] in (0, 1)
    assert response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "python", "k": 0},
        {"q": "python", "k": 1000},
        {"q": "python", "scorer": "wrong"},
        {"q": "python", "page": 0},
    ],
)
def test_invalid_search_validation(
    client: TestClient, params: dict[str, object]
) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_bad_boolean_query(client: TestClient) -> None:
    response = client.get("/search", params={"q": "python AND"})
    assert response.status_code == 422


def test_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "python", "k": 1, "page": 1}).json()
    second = client.get("/search", params={"q": "python", "k": 1, "page": 2}).json()
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]


def test_not_found(client: TestClient) -> None:
    assert client.get("/docs/999").status_code == 404
    assert client.get("/view/999").status_code == 404


def test_document_and_stats(client: TestClient) -> None:
    assert client.get("/docs/0").json()["path"] == "first.txt"
    assert client.get("/stats").json()["documents"] == 2
    assert client.get("/view/0").status_code == 200


def test_health_without_index(client: TestClient) -> None:
    assert client.get("/health").status_code == 503


def test_health_with_loaded_json(tmp_path: Path) -> None:
    index = Index(
        postings={"python": [Posting(0, 1)]},
        documents={0: DocMeta(0, "doc.txt", 1, "python")},
    )
    file = tmp_path / "index.json"
    save_index(index, file, "json")
    application = create_app(Settings(index_path=file))
    with TestClient(application) as http:
        assert http.get("/health").json() == {"status": "ok", "documents": 1}
        assert http.get("/search", params={"q": "python"}).status_code == 200
    assert application.state.service is None


def test_html_escapes_untrusted_text() -> None:
    index = Index(
        postings={"python": [Posting(0, 1)]},
        documents={
            0: DocMeta(0, "bad.html", 1, '<script>alert("XSS")</script> Python'),
        },
    )
    application = create_app(Settings(index_path=Path("/absent/findex-lab07.json")))
    application.dependency_overrides[get_service] = lambda: SearchService(index)
    with TestClient(application) as http:
        page = http.get("/", params={"q": "python"})
        assert page.status_code == 200
        assert '<script>alert("XSS")</script>' not in page.text
        assert "&lt;script&gt;" in page.text


def test_bench_routes_only_when_enabled(
    client: TestClient, service: SearchService
) -> None:
    assert client.get("/_bench/sync", params={"q": "python"}).status_code == 404
    application = create_app(
        Settings(index_path=Path("/absent/findex-lab07.json"), enable_bench=True)
    )
    application.dependency_overrides[get_service] = lambda: service
    with TestClient(application) as http:
        a = http.get("/_bench/async", params={"q": "python"})
        b = http.get("/_bench/sync", params={"q": "python"})
        assert a.status_code == b.status_code == 200
        assert a.json()["total"] == b.json()["total"] == 2


def test_unexpected_error_is_sanitized() -> None:
    application = create_app(Settings(index_path=Path("/absent/findex-lab07.json")))

    def explode() -> SearchService:
        raise RuntimeError("secret traceback must not leave server")

    application.dependency_overrides[get_service] = explode
    with TestClient(application, raise_server_exceptions=False) as http:
        response = http.get("/search", params={"q": "python"})
        assert response.status_code == 500
        assert "secret traceback" not in response.text
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
