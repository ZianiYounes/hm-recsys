"""CSV bruts H&M -> parquet compacts avec identifiants entiers contigus.

Sorties (data/processed/) :
  articles.parquet      une ligne par article, article_idx = 0..N-1, texte + chemin image
  customers.parquet     une ligne par client, customer_idx = 0..M-1
  transactions.parquet  31M lignes : t_dat, day, week, customer_idx, article_idx, price, channel
  meta.json             tailles, dates min/max
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from recsys.config import Paths
from recsys.utils.io import write_json, write_parquet
from recsys.utils.timer import timer

# Champs texte concatenes pour les embeddings (Phase 3)
TEXT_COLS = [
    "prod_name",
    "product_type_name",
    "colour_group_name",
    "graphical_appearance_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]


# --------------------------------------------------------------------------- articles
def build_articles(raw_dir: Path, images_dir: Path) -> pl.DataFrame:
    df = pl.read_csv(
        raw_dir / "articles.csv",
        schema_overrides={"article_id": pl.String, "detail_desc": pl.String},
        infer_schema_length=50_000,
    )
    df = (
        df.rename({"article_id": "article_id_str"})
        .with_columns(pl.col("article_id_str").cast(pl.Int32).alias("article_id"))
        .sort("article_id")
        .with_row_index("article_idx")
        .with_columns(pl.col("article_idx").cast(pl.Int32))
    )

    # Texte descriptif (les colonnes nulles sont ignorees)
    df = df.with_columns(
        pl.concat_str(
            [pl.col(c).cast(pl.String) for c in TEXT_COLS], separator=". ", ignore_nulls=True
        ).alias("text")
    )

    # Chemin image relatif : images/010/0108775015.jpg
    df = df.with_columns(
        pl.concat_str(
            [pl.col("article_id_str").str.slice(0, 3), pl.lit("/"), pl.col("article_id_str"),
             pl.lit(".jpg")]
        ).alias("image_path")
    )
    if images_dir.exists():
        stems = [p.stem for p in images_dir.rglob("*.jpg")]
        df = df.with_columns(pl.col("article_id_str").is_in(stems).alias("has_image"))
        logger.info(f"Images trouvees : {df['has_image'].sum():,} / {df.height:,}")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("has_image"))
        logger.info("Dossier images absent : has_image laisse a null (Phase 3)")

    # Colonnes numeriques en Int32 (gain memoire)
    int_cols = [c for c, t in df.schema.items() if t == pl.Int64]
    df = df.with_columns([pl.col(c).cast(pl.Int32) for c in int_cols])

    first = ["article_idx", "article_id", "article_id_str"]
    return df.select(first + [c for c in df.columns if c not in first])


# --------------------------------------------------------------------------- customers
def build_customers(raw_dir: Path) -> pl.DataFrame:
    df = pl.read_csv(raw_dir / "customers.csv", infer_schema_length=100_000)
    df = df.with_columns(
        pl.col("FN").fill_null(0).cast(pl.Int8),
        pl.col("Active").fill_null(0).cast(pl.Int8),
        pl.col("club_member_status").fill_null("UNKNOWN"),
        # Valeurs brutes : "NONE", "None", "Regularly", "Monthly", null
        pl.col("fashion_news_frequency").str.to_uppercase().fill_null("NONE"),
        pl.col("age").cast(pl.Int16),
    )
    df = (
        df.sort("customer_id")
        .with_row_index("customer_idx")
        .with_columns(pl.col("customer_idx").cast(pl.Int32))
    )
    return df


# --------------------------------------------------------------------------- transactions
def build_transactions(
    raw_dir: Path, customers: pl.DataFrame, articles: pl.DataFrame
) -> tuple[pl.DataFrame, dict]:
    tx = pl.read_csv(
        raw_dir / "transactions_train.csv",
        schema_overrides={
            "t_dat": pl.String,
            "customer_id": pl.String,
            "article_id": pl.String,
            "price": pl.Float32,
            "sales_channel_id": pl.Int8,
        },
    )
    logger.info(f"Transactions brutes : {tx.height:,}")

    tx = tx.with_columns(
        pl.col("t_dat").str.to_date("%Y-%m-%d"),
        pl.col("article_id").cast(pl.Int32),
    )

    cust_map = customers.select("customer_id", "customer_idx")
    art_map = articles.select("article_id", "article_idx")
    tx = tx.join(cust_map, on="customer_id", how="left").join(art_map, on="article_id", how="left")

    n_bad = tx.filter(pl.col("customer_idx").is_null() | pl.col("article_idx").is_null()).height
    if n_bad:
        logger.warning(f"{n_bad:,} transactions sans client ou article connu : supprimees")
        tx = tx.drop_nulls(["customer_idx", "article_idx"])

    min_date, max_date = tx["t_dat"].min(), tx["t_dat"].max()
    tx = tx.with_columns(
        (pl.col("t_dat") - pl.lit(min_date)).dt.total_days().cast(pl.Int16).alias("day"),
        # week 0 = les 7 derniers jours, week 1 = la semaine d'avant, etc.
        ((pl.lit(max_date) - pl.col("t_dat")).dt.total_days() // 7).cast(pl.Int16).alias("week"),
    )

    tx = tx.select(
        "t_dat", "day", "week", "customer_idx", "article_idx", "price", "sales_channel_id"
    ).sort(["customer_idx", "t_dat"])

    meta = {
        "n_transactions": tx.height,
        "min_date": str(min_date),
        "max_date": str(max_date),
        "n_weeks": int(tx["week"].max()) + 1,
        "n_active_customers": tx["customer_idx"].n_unique(),
        "n_sold_articles": tx["article_idx"].n_unique(),
    }
    return tx, meta


# --------------------------------------------------------------------------- pipeline
def run_preprocess(paths: Paths) -> None:
    raw, out = paths.raw, paths.processed

    with timer("articles"):
        articles = build_articles(raw, paths.images)
        write_parquet(articles, out / "articles.parquet")

    with timer("customers"):
        customers = build_customers(raw)
        write_parquet(customers, out / "customers.parquet")

    with timer("transactions"):
        tx, meta = build_transactions(raw, customers, articles)
        write_parquet(tx, out / "transactions.parquet")

    meta |= {"n_articles": articles.height, "n_customers": customers.height}
    write_json(meta, out / "meta.json")
    for k, v in meta.items():
        logger.info(f"{k:>20} : {v:,}" if isinstance(v, int) else f"{k:>20} : {v}")
    logger.success("Preprocessing termine")
