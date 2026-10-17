"""Baseline 2 : re-achat. Les clients H&M rachetent souvent ce qu'ils ont deja achete
(autre taille, autre couleur du meme modele, reassort basique).

Score = somme des achats ponderes par la recence : 1 / (jours_ecoules + 1)."""
from __future__ import annotations

import polars as pl


def repurchase_candidates(
    history: pl.LazyFrame,
    target_week: int,
    customers: pl.DataFrame,
    ref_day: int,
    n_weeks: int = 8,
    max_per_user: int = 100,
) -> pl.DataFrame:
    return (
        history.filter(pl.col("week") <= target_week + n_weeks)
        .join(customers.lazy().select("customer_idx"), on="customer_idx", how="semi")
        .with_columns((1.0 / (ref_day - pl.col("day").cast(pl.Int32) + 1)).alias("w"))
        .group_by(["customer_idx", "article_idx"])
        .agg(pl.col("w").sum().cast(pl.Float32).alias("score"))
        .sort(["customer_idx", "score"], descending=[False, True])
        .group_by("customer_idx", maintain_order=True)
        .head(max_per_user)
        .collect()
    )
