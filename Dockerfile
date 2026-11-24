# Stage 1: locked dependencies and non-editable package install.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY examples/web_demo/ ./examples/web_demo/
RUN uv sync --locked --no-dev --no-editable
# Ship a safe public demo corpus; do not upload personal/local crawled documents.
RUN .venv/bin/python -c "from pathlib import Path; from findex.index import build_index; from findex.storage import save_index; save_index(build_index(Path('examples/web_demo')),Path('/app/demo_index.json'),'json')"

# Stage 2: minimal runtime, unprivileged user, no uv or build toolchain.
FROM python:3.13-slim
WORKDIR /app
RUN groupadd --system findex && useradd --system --gid findex --home-dir /app --no-create-home findex
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/examples/web_demo /app/examples/web_demo
COPY --from=builder /app/demo_index.json /app/demo_index.json
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INDEX_PATH=/app/demo_index.json \
    ENABLE_BENCH_ENDPOINTS=0 \
    PORT=8000
USER findex
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn findex.web.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
