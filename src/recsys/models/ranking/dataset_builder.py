"""Construit la table (candidats x features x label) pour une semaine cible."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from recsys.data.split import history_for, targets_for
from recsys.features.feature_store import build_features
from recsys.models.baselines import baseline_candidates
from recsys.models.ranking.candidate_generation import generate_candidates
from recsys.utils.timer import timer


def add_labels(cands: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    positives = (
        targets.explode("purchased")
        .select(
            pl.col("customer_idx").cast(pl.Int32),
            pl.col("purchased").cast(pl.Int32).alias("article_idx"),
            pl.lit(1, dtype=pl.Int8).alias("label"),
        )
        .unique(["customer_idx", "article_idx"])
    )
    return cands.join(positives, on=["customer_idx", "article_idx"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )


def build_week_dataset(
    tx: pl.LazyFrame,
    week: int,
    retrievers: dict[str, dict[str, Any]],
    articles: pl.DataFrame,
    item_cat_df: pl.DataFrame,
    user_attr_df: pl.DataFrame,
    rank_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    clip_emb: np.ndarray | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, pl.DataFrame]]:
    """retrievers : {nom: cache du retriever pour cette semaine}.
    -> (table ranker, verite terrain, sources brutes)."""
    with timer(f"[semaine {week}] historique"):
        history = history_for(tx, week).collect()
        targets = targets_for(tx, week).collect()
    ref_day = int(history["day"].max())
    hist = history.lazy()

    base = baseline_candidates(hist, week, targets.select("customer_idx"), ref_day, base_cfg)
    sources = {
        "rep": base["re-achat"],
        **{name: r["candidates"] for name, r in retrievers.items()},
        "co": base["co-achat"],
        "pop": base["popularite"],
    }
    with timer(f"[semaine {week}] candidats"):
        cands = generate_candidates(sources, rank_cfg["candidates"])
    with timer(f"[semaine {week}] features"):
        df = build_features(
            cands, hist, week, ref_day, articles, item_cat_df, user_attr_df, retrievers,
            clip_emb=clip_emb,
        )
    df = add_labels(df, targets).sort(["customer_idx", "article_idx"])

    n_pos = int(df["label"].sum())
    n_target = int(targets["purchased"].list.len().sum())
    logger.info(
        f"[semaine {week}] {df.height:,} candidats pour {targets.height:,} clients "
        f"({df.height / targets.height:.0f}/client) | plafond de rappel : "
        f"{n_pos:,}/{n_target:,} = {100 * n_pos / n_target:.1f} %"
    )
    return df, targets, sources
