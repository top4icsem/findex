"""Lab 08 Typer subcommand: create MiniLM embeddings from a trusted index."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from findex.open_index import open_index
from findex.semantic import MODEL, make_embeddings


def embed_command(
    index_file: Annotated[Path, typer.Argument(help="Saved .bin or .json index")],
    out: Annotated[
        Path, typer.Option("--out", help="NumPy .npy embedding destination")
    ] = Path("embeddings.npy"),
    model: Annotated[str, typer.Option("--model")] = MODEL,
    chunk_size: Annotated[int, typer.Option("--chunk-size", min=16)] = 160,
    overlap: Annotated[int, typer.Option("--overlap", min=0)] = 32,
) -> None:
    """Compute L2-normalized chunk vectors; requires the semantic extra."""
    if not index_file.is_file():
        typer.echo(f"Index file not found: {index_file}", err=True)
        raise typer.Exit(code=2)
    try:
        with open_index(index_file) as index:
            chunks, dim = make_embeddings(
                index,
                out,
                model_name=model,
                chunk_size=chunk_size,
                overlap=overlap,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Embedding error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Saved {chunks} chunks x {dim} dimensions to {out}")
