"""Calcule les candidats des 3 baselines pour les clients cibles d'un split."""
from __future__ import annotations

from typing import Any

import polars as pl

from recsys.models.baselines.item_cooccurrence import cooccurrence_candidates, cooccurrence_table
from recsys.models.baselines.popularity import popular_items, popularity_candidates
from recsys.models.baselines.repurchase import repurchase_candidates
from recsys.utils.timer import timer


def baseline_candidates(
    history: pl.LazyFrame,
    target_week: int,
    customers: pl.DataFrame,
    ref_day: int,
    cfg: dict[str, Any],
) -> dict[str, pl.DataFrame]:
    """-> {"popularite", "re-achat", "co-achat"} : DataFrames (client, article, score)."""
    pc, rc, cc = cfg["popularity"], cfg["repurchase"], cfg["cooccurrence"]
    with timer("popularite"):
        pop = popularity_candidates(
            customers, popular_items(history, target_week, pc["n_weeks"], pc["top_n"])
        )
    with timer("re-achat"):
        rep = repurchase_candidates(
            history, target_week, customers, ref_day, rc["n_weeks"], rc["max_per_user"]
        )
    with timer("co-achat"):
        table = cooccurrence_table(
            history, target_week, cc["n_weeks"], cc["max_items_per_user"], cc["top_n"]
        )
        co = cooccurrence_candidates(
            history, target_week, customers, table, ref_day, cc["source_n_weeks"],
            cc["source_items"],
        )
    return {"popularite": pop, "re-achat": rep, "co-achat": co}
