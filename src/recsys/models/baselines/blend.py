"""Fusion par priorite : re-achat d'abord, puis co-achat, puis popularite pour completer."""
from __future__ import annotations

import polars as pl


def blend_by_priority(sources: list[pl.DataFrame], k: int) -> pl.DataFrame:
    """sources : DataFrames (customer_idx, article_idx, score), du plus au moins prioritaire.

    Renvoie (customer_idx, preds) sans doublon, k articles max par client."""
    parts = [
        s.select("customer_idx", "article_idx", pl.col("score").cast(pl.Float32)).with_columns(
            pl.lit(i, dtype=pl.Int8).alias("prio")
        )
        for i, s in enumerate(sources)
    ]
    return (
        pl.concat(parts)
        .sort(["customer_idx", "prio", "score"], descending=[False, False, True])
        .unique(["customer_idx", "article_idx"], keep="first", maintain_order=True)
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").head(k).alias("preds"))
    )
