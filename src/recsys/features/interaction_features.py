"""Features client x article : le client a-t-il deja achete cet article, ou ce modele ?"""
from __future__ import annotations

import polars as pl


def user_item_features(
    cands: pl.DataFrame,
    history: pl.LazyFrame,
    articles: pl.DataFrame,
    week: int,
    ref_day: int,
) -> pl.DataFrame:
    """Ajoute aux candidats (customer_idx, article_idx, ...) :
    ui_count, ui_count_4w, ui_days_since : achats passes de cet article precis
    upc_count, upc_days_since           : achats passes du meme modele (product_code),
                                          toutes couleurs confondues"""
    art_pc = articles.select("article_idx", "product_code").lazy()
    h = history.join(
        cands.lazy().select("customer_idx").unique(), on="customer_idx", how="semi"
    )

    ui = (
        h.group_by(["customer_idx", "article_idx"])
        .agg(
            pl.len().alias("ui_count"),
            (pl.col("week") <= week + 4).sum().alias("ui_count_4w"),
            (pl.lit(ref_day) - pl.col("day").max().cast(pl.Int32)).alias("ui_days_since"),
        )
        .collect()
    )
    upc = (
        h.join(art_pc, on="article_idx")
        .group_by(["customer_idx", "product_code"])
        .agg(
            pl.len().alias("upc_count"),
            (pl.lit(ref_day) - pl.col("day").max().cast(pl.Int32)).alias("upc_days_since"),
        )
        .collect()
    )
    return (
        cands.join(ui, on=["customer_idx", "article_idx"], how="left")
        .join(art_pc.collect(), on="article_idx", how="left")
        .join(upc, on=["customer_idx", "product_code"], how="left")
        .drop("product_code")
    )
