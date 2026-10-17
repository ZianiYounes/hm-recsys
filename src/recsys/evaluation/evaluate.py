"""Evaluation offline : predictions (DataFrame) vs verite terrain d'un split."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from recsys.evaluation.coverage import catalog_coverage, user_coverage
from recsys.evaluation.metrics import compute_metrics


def candidates_to_lists(cands: pl.DataFrame, k: int) -> pl.DataFrame:
    """(customer_idx, article_idx, score) -> (customer_idx, preds: liste top-k par score)."""
    return (
        cands.sort(["customer_idx", "score"], descending=[False, True])
        .unique(["customer_idx", "article_idx"], keep="first", maintain_order=True)
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").head(k).alias("preds"))
    )


def evaluate_predictions(
    preds: pl.DataFrame,
    targets: pl.DataFrame,
    ks: Sequence[int] = (12,),
    n_articles: int | None = None,
) -> dict[str, float]:
    """preds : (customer_idx, preds) ; targets : (customer_idx, purchased).

    Un client cible sans prediction compte comme un echec (score 0)."""
    df = targets.join(preds, on="customer_idx", how="left")
    p_lists = [p if p is not None else [] for p in df["preds"].to_list()]
    t_lists = df["purchased"].to_list()

    out = compute_metrics(p_lists, t_lists, ks=ks)
    out["user_coverage"] = user_coverage(p_lists)
    if n_articles:
        out["catalog_coverage_at_12"] = catalog_coverage(p_lists, n_articles, k=12)
    return out
