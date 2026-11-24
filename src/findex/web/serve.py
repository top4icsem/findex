"""Typer command to run the FastAPI service with Uvicorn."""

from __future__ import annotations

import os
from typing import Annotated

import typer
import uvicorn


def serve_command(
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
) -> None:
    """Launch the web search service (use --host 0.0.0.0 for containers)."""
    # Uvicorn's import-string mode makes --workers and spawn safe.
    # PORT is supplied by Render/Railway; explicit --port overrides it.
    selected_port = int(os.environ.get("PORT", str(port))) if port == 8000 else port
    uvicorn.run(
        "findex.web.app:app",
        host=host,
        port=selected_port,
        workers=workers,
        log_level="info",
    )
