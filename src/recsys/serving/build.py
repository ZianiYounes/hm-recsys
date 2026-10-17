"""Prepare tout ce dont l'API a besoin pour recommander la semaine qui suit les donnees.

Semaine servie = -1 (la semaine apres le 2020-09-22) :
- retrievers entraines sur tout l'historique ;
- ranker entraine sur les semaines les plus recentes (0 et 1) ;
- tables globales precalculees (popularite, co-achat, stats articles)."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from recsys.config import Paths, load_config
from recsys.features.item_features import item_stats
from recsys.models.baselines import cooccurrence_table, popular_items
from recsys.training.train_ranker import train_final_ranker
from recsys.training.train_two_tower import fit_two_tower, save_checkpoint
from recsys.utils.io import write_json, write_parquet
from recsys.utils.timer import timer

SERVE_WEEK = -1


def build_serving(
    paths: Paths,
    rank_cfg: dict[str, Any] | None = None,
    retriever_cfgs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rank_cfg = rank_cfg or load_config("ranker")
    retriever_cfgs = retriever_cfgs or {r: load_config(r) for r in rank_cfg["retrievers"]}
    base_cfg = load_config("baselines")
    out = paths.artifacts / "serving"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Ranker de production (reutilise les caches des retrievers par semaine)
    features = train_final_ranker(paths, rank_cfg, SERVE_WEEK, out, retriever_cfgs)

    # 2. Retrievers entraines sur tout l'historique
    history = pl.read_parquet(paths.processed / "transactions.parquet")
    ref_day = int(history["day"].max())
    for name, rcfg in retriever_cfgs.items():
        with timer(f"{name} (production)"):
            res = fit_two_tower(paths, SERVE_WEEK, history, None, rcfg)
        save_checkpoint(out / f"{name}.pt", res["model"], rcfg)
        np.save(out / f"{name}_item_emb.npy", res["item_emb"].astype(np.float32))
        np.save(out / f"{name}_allowed.npy", res["allowed_ids"])

    # 3. Tables globales
    hist = history.lazy()
    pc, cc = base_cfg["popularity"], base_cfg["cooccurrence"]
    write_parquet(
        popular_items(hist, SERVE_WEEK, pc["n_weeks"], rank_cfg["candidates"]["pop"]),
        out / "popular.parquet",
    )
    write_parquet(
        cooccurrence_table(
            hist, SERVE_WEEK, cc["n_weeks"], cc["max_items_per_user"], cc["top_n"]
        ),
        out / "cooccurrence.parquet",
    )
    write_parquet(item_stats(hist, SERVE_WEEK, ref_day), out / "item_stats.parquet")

    meta = {
        "serve_week": SERVE_WEEK,
        "ref_day": ref_day,
        "reference_date": str(history["t_dat"].max()),
        "retrievers": list(retriever_cfgs),
        "use_clip_features": bool(rank_cfg.get("use_clip_features", False)),
        "features": features,
        "candidates": rank_cfg["candidates"],
        "baselines": base_cfg,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(meta, out / "meta.json")
    logger.success(f"Artefacts de serving prets dans {out}")
    return meta
