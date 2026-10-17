"""Baseline 1 : les articles les plus vendus sur les dernieres semaines de l'historique.

Meme liste pour tout le monde : c'est le filet de securite pour les clients froids."""
from __future__ import annotations

import polars as pl


def popular_items(
    history: pl.LazyFrame, target_week: int, n_weeks: int = 1, top_n: int = 100
) -> pl.DataFrame:
    """(article_idx, score) : ventes sur les n_weeks semaines precedant la semaine cible."""
    return (
        history.filter(pl.col("week") <= target_week + n_weeks)
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Float32).alias("score"))
        .sort(["score", "article_idx"], descending=[True, False])
        .head(top_n)
        .collect()
    )


def popularity_candidates(customers: pl.DataFrame, popular: pl.DataFrame) -> pl.DataFrame:
    """Diffuse la liste populaire a chaque client cible."""
    return customers.select("customer_idx").join(popular, how="cross")
