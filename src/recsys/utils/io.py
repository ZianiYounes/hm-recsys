from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=True)
    size_mb = path.stat().st_size / 1024**2
    logger.info(f"Ecrit {path.name} : {df.height:,} lignes, {size_mb:.1f} Mo")


def read_parquet(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(path, columns=columns)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
