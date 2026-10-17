"""Split temporel sans fuite de donnees.

Pour une semaine cible W :
  historique = toutes les transactions strictement anterieures (week > W)
  cibles     = pour chaque client ayant achete en semaine W, la liste de ses articles

val  : W = 1  (entrainement sur week >= 2, reglage des hyperparametres)
test : W = 0  (entrainement sur week >= 1, score final reporte dans le README)
"""
from __future__ import annotations

import polars as pl
from loguru import logger

from recsys.config import Paths
from recsys.utils.io import write_json, write_parquet

FrameLike = pl.DataFrame | pl.LazyFrame


def history_for(tx: FrameLike, target_week: int, history_weeks: int | None = None) -> pl.LazyFrame:
    """Transactions visibles avant la semaine cible (aucune fuite du futur)."""
    hist = tx.lazy().filter(pl.col("week") > target_week)
    if history_weeks is not None:
        hist = hist.filter(pl.col("week") <= target_week + history_weeks)
    return hist


def targets_for(tx: FrameLike, target_week: int) -> pl.LazyFrame:
    """Verite terrain : articles uniques achetes par client pendant la semaine cible."""
    return (
        tx.lazy()
        .filter(pl.col("week") == target_week)
        .sort(["customer_idx", "t_dat"])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").unique(maintain_order=True).alias("purchased"))
    )


def _split_stats(history: pl.LazyFrame, targets: pl.DataFrame) -> dict:
    seen = history.select("customer_idx").unique().collect()
    n_with_hist = targets.join(seen, on="customer_idx", how="semi").height
    return {
        "n_target_customers": targets.height,
        "avg_items_per_customer": round(targets["purchased"].list.len().mean(), 3),
        "pct_customers_with_history": round(100 * n_with_hist / max(targets.height, 1), 2),
        "n_history_rows": history.select(pl.len()).collect().item(),
    }


def build_and_save_splits(
    paths: Paths, val_week: int = 1, test_week: int = 0, history_weeks: int | None = None
) -> dict:
    tx = pl.scan_parquet(paths.processed / "transactions.parquet")
    info: dict = {"history_weeks": history_weeks}

    for name, week in [("val", val_week), ("test", test_week)]:
        targets = targets_for(tx, week).collect()
        write_parquet(targets, paths.splits / f"{name}_targets.parquet")
        stats = _split_stats(history_for(tx, week, history_weeks), targets)
        info[name] = {"target_week": week, **stats}
        logger.info(f"[{name}] semaine {week} : {stats}")

    write_json(info, paths.splits / "split_info.json")
    logger.success("Splits enregistres dans " + str(paths.splits))
    return info


def target_week_of(paths: Paths, name: str) -> int:
    import json

    info = json.loads((paths.splits / "split_info.json").read_text(encoding="utf-8"))
    return int(info[name]["target_week"])


def load_split(paths: Paths, name: str) -> tuple[pl.LazyFrame, pl.DataFrame]:
    """Renvoie (historique lazy, cibles) pour 'val' ou 'test'."""
    import json

    info = json.loads((paths.splits / "split_info.json").read_text(encoding="utf-8"))
    tx = pl.scan_parquet(paths.processed / "transactions.parquet")
    history = history_for(tx, info[name]["target_week"], info["history_weeks"])
    targets = pl.read_parquet(paths.splits / f"{name}_targets.parquet")
    return history, targets
