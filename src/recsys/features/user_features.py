"""Features clients categorielles -> codes entiers (age par tranches de 5 ans, statut, news)."""
from __future__ import annotations

import numpy as np
import polars as pl

USER_FEATURES = ["age_bucket", "FN", "Active", "club_member_status", "fashion_news_frequency"]


def encode_user_features(customers: pl.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Renvoie (codes [n_customers, 5] int64 ordonnes par customer_idx, cardinalites)."""
    c = customers.sort("customer_idx").with_columns(
        # 0 = age inconnu, puis tranches de 5 ans a partir de 15 ans
        pl.when(pl.col("age").is_null())
        .then(0)
        .otherwise(((pl.col("age") - 15).clip(0, 70) // 5) + 1)
        .cast(pl.Int64)
        .alias("age_bucket"),
        (pl.col("club_member_status").rank("dense") - 1).cast(pl.Int64),
        (pl.col("fashion_news_frequency").rank("dense") - 1).cast(pl.Int64),
        pl.col("FN").cast(pl.Int64),
        pl.col("Active").cast(pl.Int64),
    )
    codes = np.stack([c[col].to_numpy() for col in USER_FEATURES], axis=1)
    cards = [int(codes[:, i].max()) + 1 for i in range(codes.shape[1])]
    return codes, cards


def user_stats(
    history: pl.LazyFrame, customers: pl.DataFrame, week: int, ref_day: int
) -> pl.DataFrame:
    """Comportement d'achat des clients cibles, calcule sur l'historique (semaines > week)."""
    return (
        history.join(customers.lazy().select("customer_idx"), on="customer_idx", how="semi")
        .group_by("customer_idx")
        .agg(
            pl.len().alias("user_n_purchases"),
            (pl.col("week") <= week + 4).sum().alias("user_n_4w"),
            (pl.lit(ref_day) - pl.col("day").max().cast(pl.Int32)).alias("user_days_since"),
            pl.col("price").mean().alias("user_mean_price"),
            pl.col("price").max().alias("user_max_price"),
            (pl.col("sales_channel_id") == 2).mean().alias("user_online_share"),
            pl.col("article_idx").n_unique().alias("user_n_unique_items"),
        )
        .collect()
    )


def user_attribute_frame(customers: pl.DataFrame) -> pl.DataFrame:
    """Attributs statiques des clients pour LightGBM."""
    codes, _ = encode_user_features(customers)
    ages = customers.sort("customer_idx")["age"].cast(pl.Float32)
    return pl.DataFrame(
        {
            "customer_idx": np.arange(len(codes), dtype=np.int32),
            "age": ages,
            "fn": codes[:, 1],
            "active": codes[:, 2],
            "cat_club": codes[:, 3],
            "cat_news": codes[:, 4],
        }
    )
