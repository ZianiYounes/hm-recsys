"""Attributs categoriels des articles -> codes entiers denses (pour nn.Embedding)."""
from __future__ import annotations

import numpy as np
import polars as pl

ITEM_CAT_COLS = [
    "product_code",
    "product_type_no",
    "product_group_name",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_code",
    "index_group_no",
    "section_no",
    "garment_group_no",
]


def encode_item_categoricals(
    articles: pl.DataFrame, cols: list[str] = ITEM_CAT_COLS
) -> tuple[np.ndarray, list[int]]:
    """Renvoie (codes [n_articles, n_cols] int64 ordonnes par article_idx, cardinalites)."""
    arts = articles.sort("article_idx")
    codes, cards = [], []
    for c in cols:
        col = arts[c].cast(pl.String).fill_null("NA")
        dense = (col.rank("dense") - 1).cast(pl.Int64).to_numpy()
        codes.append(dense)
        cards.append(int(dense.max()) + 1)
    return np.stack(codes, axis=1), cards


def item_stats(history: pl.LazyFrame, week: int, ref_day: int) -> pl.DataFrame:
    """Statistiques de vente par article, calculees sur l'historique (semaines > week)."""
    last_4w = pl.col("week") <= week + 4
    return (
        history.group_by("article_idx")
        .agg(
            pl.len().alias("item_sales_all"),
            (pl.col("week") == week + 1).sum().alias("item_sales_1w"),
            last_4w.sum().alias("item_sales_4w"),
            pl.col("customer_idx").filter(last_4w).n_unique().alias("item_buyers_4w"),
            pl.col("price").filter(last_4w).mean().alias("item_price_4w"),
            pl.col("price").mean().alias("item_price_all"),
            (pl.lit(ref_day) - pl.col("day").min().cast(pl.Int32)).alias("item_age_days"),
            (pl.lit(ref_day) - pl.col("day").max().cast(pl.Int32)).alias("item_days_since_sale"),
            (pl.col("sales_channel_id") == 2).mean().alias("item_online_share"),
        )
        .with_columns(
            (pl.col("item_sales_1w") / (pl.col("item_sales_4w") / 4 + 1)).alias("item_trend"),
            (pl.col("item_price_4w") / pl.col("item_price_all")).alias("item_price_drop"),
        )
        .collect()
    )


def item_categorical_frame(articles: pl.DataFrame) -> pl.DataFrame:
    """Codes categoriels pour LightGBM (sans product_code, trop de modalites)."""
    codes, _ = encode_item_categoricals(articles)
    cols = {
        f"cat_{c}": codes[:, i] for i, c in enumerate(ITEM_CAT_COLS) if c != "product_code"
    }
    return pl.DataFrame({"article_idx": np.arange(len(codes), dtype=np.int32), **cols})
