"""Union des candidats de toutes les sources, avec score et rang de chaque source.

Un candidat absent d'une source a score et rang nuls (NaN) : LightGBM sait le gerer."""
from __future__ import annotations

import polars as pl


def top_per_user(df: pl.DataFrame, n: int, prefix: str) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("score")
            .rank("ordinal", descending=True)
            .over("customer_idx")
            .cast(pl.Int32)
            .alias(f"{prefix}_rank")
        )
        .filter(pl.col(f"{prefix}_rank") <= n)
        .select(
            pl.col("customer_idx").cast(pl.Int32),
            pl.col("article_idx").cast(pl.Int32),
            pl.col("score").cast(pl.Float32).alias(f"{prefix}_score"),
            f"{prefix}_rank",
        )
    )


def generate_candidates(
    sources: dict[str, pl.DataFrame], n_per_source: dict[str, int]
) -> pl.DataFrame:
    """sources : {"rep": df, "tt": df, ...}, chaque df = (customer_idx, article_idx, score)."""
    out: pl.DataFrame | None = None
    for prefix, df in sources.items():
        part = top_per_user(df, n_per_source[prefix], prefix)
        out = part if out is None else out.join(
            part, on=["customer_idx", "article_idx"], how="full", coalesce=True
        )
    assert out is not None
    return out.with_columns(
        pl.sum_horizontal(
            [pl.col(f"{p}_rank").is_not_null().cast(pl.Int8) for p in sources]
        ).alias("n_sources")
    ).sort(["customer_idx", "article_idx"])
