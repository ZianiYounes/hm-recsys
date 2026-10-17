"""Assemble toutes les features du ranker pour une table de candidats.

Le meme code sert a l'entrainement (millions de lignes) et au serving (un seul client) :
pas de divergence entre les features vues a l'entrainement et en production."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch

from recsys.features.interaction_features import user_item_features
from recsys.features.item_features import ITEM_CAT_COLS, item_stats
from recsys.features.user_features import user_stats

ITEM_CAT_FEATURES = [f"cat_{c}" for c in ITEM_CAT_COLS if c != "product_code"]
CATEGORICAL_FEATURES = ITEM_CAT_FEATURES + ["cat_club", "cat_news"]

BASE_FEATURES = (
    ["ui_count", "ui_count_4w", "ui_days_since", "upc_count", "upc_days_since"]
    + [
        "item_sales_all", "item_sales_1w", "item_sales_4w", "item_buyers_4w",
        "item_price_4w", "item_price_all", "item_age_days", "item_days_since_sale",
        "item_online_share", "item_trend", "item_price_drop",
    ]
    + [
        "user_n_purchases", "user_n_4w", "user_days_since", "user_mean_price",
        "user_max_price", "user_online_share", "user_n_unique_items",
    ]
    + ["age", "fn", "active", "price_ratio"]
)


def source_names(retrievers: list[str]) -> list[str]:
    return ["rep", *retrievers, "co", "pop"]


def feature_columns(retrievers: list[str], use_clip: bool = False) -> list[str]:
    sources = source_names(retrievers)
    return (
        [f"{s}_{k}" for s in sources for k in ("score", "rank")]
        + ["n_sources"]
        + BASE_FEATURES
        + [f"{r}_sim" for r in retrievers]
        + (["clip_sim"] if use_clip else [])
        + CATEGORICAL_FEATURES
    )


def embedding_similarity(
    customers: np.ndarray,
    articles: np.ndarray,
    user_ids: np.ndarray,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    chunk: int = 500_000,
) -> np.ndarray:
    """Produit scalaire client x article pour chaque paire (NaN si client inconnu)."""
    out = np.full(len(customers), np.nan, dtype=np.float32)
    if len(customers) == 0 or len(user_ids) == 0:
        return out
    row_of = np.full(int(max(customers.max(), user_ids.max())) + 1, -1, dtype=np.int64)
    row_of[user_ids] = np.arange(len(user_ids))
    rows = row_of[customers]
    for s in range(0, len(customers), chunk):
        r, a = rows[s : s + chunk], articles[s : s + chunk]
        ok = r >= 0
        out[s : s + chunk][ok] = np.einsum(
            "ij,ij->i", user_emb[r[ok]].astype(np.float32), item_emb[a[ok]].astype(np.float32)
        )
    return out


def two_tower_similarity(customers: np.ndarray, articles: np.ndarray, tt: dict[str, Any]):
    return embedding_similarity(customers, articles, tt["user_ids"], tt["user_emb"], tt["item_emb"])


def clip_profiles(
    history: pl.LazyFrame, customers: pl.DataFrame, clip_emb: np.ndarray, last_n: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Profil visuel du client = moyenne FashionCLIP de ses `last_n` derniers articles."""
    recent = (
        history.join(customers.lazy(), on="customer_idx", how="semi")
        .group_by(["customer_idx", "article_idx"])
        .agg(pl.col("day").max())
        .sort(["customer_idx", "day"], descending=[False, True])
        .group_by("customer_idx", maintain_order=True)
        .head(last_n)
        .collect()
    )
    user_ids = recent["customer_idx"].unique(maintain_order=True).to_numpy()
    if len(user_ids) == 0:
        return user_ids.astype(np.int32), np.zeros((0, clip_emb.shape[1]), dtype=np.float32)
    rows = recent["customer_idx"].rank("dense").cast(pl.Int64).to_numpy() - 1
    emb = torch.from_numpy(clip_emb)
    prof = torch.zeros(len(user_ids), clip_emb.shape[1])
    items = recent["article_idx"].to_numpy()
    for s in range(0, len(items), 200_000):
        idx = torch.from_numpy(rows[s : s + 200_000])
        prof.index_add_(0, idx, emb[items[s : s + 200_000]].float())
    prof = torch.nn.functional.normalize(prof, dim=1).numpy()
    # rank("dense") suit l'ordre croissant des ids : on aligne user_ids dessus
    return np.sort(user_ids).astype(np.int32), prof


def build_features(
    cands: pl.DataFrame,
    history: pl.LazyFrame,
    week: int,
    ref_day: int,
    articles: pl.DataFrame,
    item_cat_df: pl.DataFrame,
    user_attr_df: pl.DataFrame,
    retrievers: dict[str, dict[str, Any]],
    item_stats_df: pl.DataFrame | None = None,
    clip_emb: np.ndarray | None = None,
) -> pl.DataFrame:
    """retrievers : {nom: {"user_ids", "user_emb", "item_emb"}}.
    item_stats_df : stats articles precalculees (serving), sinon calculees sur `history`."""
    customers = cands.select("customer_idx").unique()
    if item_stats_df is None:
        item_stats_df = item_stats(history, week, ref_day)
    df = user_item_features(cands, history, articles, week, ref_day)
    df = df.join(item_stats_df, on="article_idx", how="left")
    df = df.join(user_stats(history, customers, week, ref_day), on="customer_idx", how="left")
    df = df.join(item_cat_df, on="article_idx", how="left")
    df = df.join(user_attr_df, on="customer_idx", how="left")

    cust = df["customer_idx"].to_numpy()
    arts = df["article_idx"].to_numpy()
    sims = [
        pl.Series(f"{name}_sim", two_tower_similarity(cust, arts, r))
        for name, r in retrievers.items()
    ]
    if clip_emb is not None:
        uids, prof = clip_profiles(history, customers, clip_emb)
        sims.append(pl.Series("clip_sim", embedding_similarity(cust, arts, uids, prof, clip_emb)))
    return df.with_columns(
        (pl.col("item_price_4w") / pl.col("user_mean_price")).alias("price_ratio"), *sims
    )
