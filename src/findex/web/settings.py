"""Runtime configuration, read only from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    index_path: Path = Path("index.bin")
    enable_bench: bool = False
    embeddings_path: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            index_path=Path(os.environ.get("INDEX_PATH", "index.bin")),
            embeddings_path=(
                Path(os.environ["EMBEDDINGS_PATH"])
                if os.environ.get("EMBEDDINGS_PATH")
                else None
            ),
            enable_bench=os.environ.get("ENABLE_BENCH_ENDPOINTS", "0").lower()
            in {"1", "true", "yes"},
        )
