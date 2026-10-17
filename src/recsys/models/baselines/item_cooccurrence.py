"""Baseline 3 : co-achat ("les clients qui ont achete A ont aussi achete B").

1. table de co-occurrence A -> B sur les achats recents
2. pour chaque client : voisins de ses derniers articles, ponderes par la recence"""
from __future__ import annotations

import polars as pl


def _recent_unique_items(
    history: pl.LazyFrame, target_week: int, n_weeks: int, max_items: int
) -> pl.LazyFrame:
    """Articles uniques par client, du plus recent au plus ancien, plafonnes a max_items."""
    return (
        history.filter(pl.col("week") <= target_week + n_weeks)
        .group_by(["customer_idx", "article_idx"])
        .agg(pl.col("day").max())
        .sort(["customer_idx", "day"], descending=[False, True])
        .group_by("customer_idx", maintain_order=True)
        .head(max_items)
    )


def cooccurrence_table(
    history: pl.LazyFrame,
    target_week: int,
    n_weeks: int = 4,
    max_items_per_user: int = 30,
    top_n: int = 20,
) -> pl.DataFrame:
    """(article_idx, pair_idx, co) : top_n voisins de chaque article."""
    items = _recent_unique_items(history, target_week, n_weeks, max_items_per_user).select(
        "customer_idx", "article_idx"
    )
    pairs = (
        items.join(items.rename({"article_idx": "pair_idx"}), on="customer_idx")
        .filter(pl.col("article_idx") != pl.col("pair_idx"))
        .group_by(["article_idx", "pair_idx"])
        .agg(pl.len().cast(pl.Float32).alias("co"))
    )
    return (
        pairs.sort(["article_idx", "co"], descending=[False, True])
        .group_by("article_idx", maintain_order=True)
        .head(top_n)
        .collect()
    )


def cooccurrence_candidates(
    history: pl.LazyFrame,
    target_week: int,
    customers: pl.DataFrame,
    table: pl.DataFrame,
    ref_day: int,
    n_weeks: int = 8,
    source_items: int = 5,
) -> pl.DataFrame:
    """Score(client, B) = somme sur ses articles recents A de co(A, B) * recence(A)."""
    src = (
        _recent_unique_items(
            history.join(customers.lazy().select("customer_idx"), on="customer_idx", how="semi"),
            target_week,
            n_weeks,
            source_items,
        )
        .with_columns((1.0 / (ref_day - pl.col("day").cast(pl.Int32) + 1)).alias("w"))
        .select("customer_idx", "article_idx", "w")
    )
    return (
        src.join(table.lazy(), on="article_idx")
        .group_by(["customer_idx", "pair_idx"])
        .agg((pl.col("co") * pl.col("w")).sum().cast(pl.Float32).alias("score"))
        .rename({"pair_idx": "article_idx"})
        .collect()
    )
