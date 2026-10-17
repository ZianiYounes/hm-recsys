"""Moteur de recommandation temps reel : un client -> ses 12 articles pour la semaine suivante.

Pipeline par requete (memes fonctions qu'a l'entrainement, pas de divergence) :
  1. candidats : re-achat, co-achat, popularite + retrievers neuronaux via FAISS
  2. features  : build_features sur l'historique du seul client
  3. ranking   : LightGBM LambdaRank, puis explication de chaque recommandation"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import faiss
import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from loguru import logger

from recsys.config import Paths, load_config
from recsys.data.datasets import GpuHistory, SequenceStore
from recsys.embeddings import load_item_embeddings
from recsys.features.feature_store import build_features
from recsys.features.item_features import item_categorical_frame
from recsys.features.user_features import user_attribute_frame
from recsys.models.baselines import (
    cooccurrence_candidates,
    popularity_candidates,
    repurchase_candidates,
)
from recsys.models.ranking.candidate_generation import generate_candidates
from recsys.models.ranking.ranker import to_matrix
from recsys.models.two_tower import TwoTower
from recsys.models.two_tower.factory import build_two_tower
from recsys.retrieval.faiss_index import build_hnsw_index, search
from recsys.utils.io import read_json
from recsys.utils.timer import timer

ARTICLE_INFO_COLS = [
    "article_idx", "article_id_str", "prod_name", "product_type_name", "colour_group_name",
    "department_name", "index_group_name", "image_path",
]

RETRIEVER_REASONS = {
    "two_tower": "Matches your style",
    "sasrec": "Follows your recent purchase sequence",
}


@dataclass
class Retriever:
    model: TwoTower
    max_len: int
    item_emb: np.ndarray   # [n_items, D]
    allowed: np.ndarray    # article_idx indexes dans FAISS
    index: faiss.Index


class Recommender:
    def __init__(self, paths: Paths):
        sdir = paths.artifacts / "serving"
        if not (sdir / "meta.json").exists():
            raise FileNotFoundError(
                f"{sdir / 'meta.json'} introuvable : lance `python scripts/build_serving.py`"
            )
        self.meta: dict[str, Any] = read_json(sdir / "meta.json")
        scfg = load_config("serving")
        torch.set_num_threads(scfg.get("torch_threads", 4))
        self.week, self.ref_day = self.meta["serve_week"], self.meta["ref_day"]

        with timer("chargement des donnees"):
            self.articles = pl.read_parquet(paths.processed / "articles.parquet")
            customers = pl.read_parquet(paths.processed / "customers.parquet")
            self.tx = pl.read_parquet(paths.processed / "transactions.parquet")
        self.article_info = self.articles.select(ARTICLE_INFO_COLS)
        self.customer_ids = customers.sort("customer_idx")["customer_id"]
        n_customers = customers.height
        counts = np.bincount(self.tx["customer_idx"].to_numpy(), minlength=n_customers)
        self.offsets = np.zeros(n_customers + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self.sample_pool = np.where(counts >= scfg.get("sample_min_purchases", 5))[0]

        self.item_cat_df = item_categorical_frame(self.articles)
        self.user_attr_df = user_attribute_frame(customers)
        self.item_stats = pl.read_parquet(sdir / "item_stats.parquet")
        self.popular = pl.read_parquet(sdir / "popular.parquet")
        self.cooc = pl.read_parquet(sdir / "cooccurrence.parquet")
        self.clip = (
            load_item_embeddings(paths) if self.meta.get("use_clip_features") else None
        )

        with timer("historiques en memoire"):
            store = SequenceStore.from_history(self.tx, n_customers)
            self.hist = GpuHistory(store, torch.device("cpu"))
        self.retrievers: dict[str, Retriever] = {}
        self.user_feats: torch.Tensor | None = None
        for name in self.meta["retrievers"]:
            with timer(f"retriever {name} + index FAISS"):
                self.retrievers[name] = self._load_retriever(paths, sdir, name, customers)

        self.booster = lgb.Booster(model_str=(sdir / "ranker.txt").read_text(encoding="utf-8"))
        self.features: list[str] = self.meta["features"]
        logger.success(
            f"Recommender pret : {n_customers:,} clients, {self.articles.height:,} articles, "
            f"retrievers {list(self.retrievers)}"
        )

    def _load_retriever(self, paths: Paths, sdir, name: str, customers: pl.DataFrame) -> Retriever:
        ckpt = torch.load(sdir / f"{name}.pt", map_location="cpu")
        model, user_feats = build_two_tower(paths, ckpt["cfg"], self.articles, customers)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        if self.user_feats is None:
            self.user_feats = torch.from_numpy(user_feats)
        item_emb = np.load(sdir / f"{name}_item_emb.npy").astype(np.float32)
        allowed = np.load(sdir / f"{name}_allowed.npy")
        return Retriever(
            model=model,
            max_len=ckpt["cfg"]["data"]["max_history"],
            item_emb=item_emb,
            allowed=allowed,
            index=build_hnsw_index(item_emb[allowed]),
        )

    # ------------------------------------------------------------------ lookups
    @property
    def reference_date(self) -> str:
        return self.meta["reference_date"]

    def customer_idx(self, customer_id: str) -> int | None:
        pos = int(self.customer_ids.search_sorted(customer_id))
        if pos < len(self.customer_ids) and self.customer_ids[pos] == customer_id:
            return pos
        return None

    def sample_customers(self, n: int, seed: int | None = None) -> list[str]:
        rng = np.random.default_rng(seed)
        picks = rng.choice(self.sample_pool, size=min(n, len(self.sample_pool)), replace=False)
        return [self.customer_ids[int(i)] for i in picks]

    def history(self, c: int) -> pl.DataFrame:
        start, end = int(self.offsets[c]), int(self.offsets[c + 1])
        return self.tx.slice(start, end - start)

    def purchases(self, c: int, limit: int) -> list[dict[str, Any]]:
        h = (
            self.history(c)
            .sort("t_dat", descending=True)
            .unique(["article_idx"], keep="first", maintain_order=True)
            .head(limit)
            .join(self.article_info, on="article_idx", how="left")
        )
        return [
            {**self._article(row), "date": str(row["t_dat"]), "price": float(row["price"])}
            for row in h.iter_rows(named=True)
        ]

    def article_idx(self, article_id: str) -> int | None:
        hit = self.article_info.filter(pl.col("article_id_str") == article_id.zfill(10))
        return int(hit["article_idx"][0]) if hit.height else None

    def article(self, a: int) -> dict[str, Any]:
        row = self.article_info.filter(pl.col("article_idx") == a).row(0, named=True)
        return self._article(row)

    @staticmethod
    def _article(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "article_id": row["article_id_str"],
            "name": row["prod_name"],
            "product_type": row["product_type_name"],
            "colour": row["colour_group_name"],
            "department": row["department_name"],
            "index_group": row["index_group_name"],
            "image_path": row["image_path"],
        }

    # ------------------------------------------------------------------ recommandation
    @torch.inference_mode()
    def _retrieve(self, name: str, c: int, k: int) -> tuple[pl.DataFrame, np.ndarray]:
        r = self.retrievers[name]
        items, gaps, mask = self.hist.last_purchases(torch.tensor([c]), self.ref_day + 1, r.max_len)
        u = r.model.encode_users(items, gaps, mask, self.user_feats[[c]]).float().numpy()
        scores, pos = search(r.index, u, k)
        ok = pos[0] >= 0
        df = pl.DataFrame(
            {
                "customer_idx": np.full(int(ok.sum()), c, dtype=np.int32),
                "article_idx": r.allowed[pos[0][ok]].astype(np.int32),
                "score": scores[0][ok].astype(np.float32),
            }
        )
        return df, u

    def _reasons(self, row: dict[str, Any]) -> list[str]:
        reasons = []
        if row.get("ui_count"):
            reasons.append("You bought this item before")
        elif row.get("upc_count"):
            reasons.append("Same model as a past purchase, another colour")
        if row.get("co_rank") is not None:
            reasons.append("Often bought with your recent items")
        for name in self.retrievers:
            if row.get(f"{name}_rank") is not None:
                reasons.append(RETRIEVER_REASONS.get(name, "Similar to your purchases"))
                break
        if row.get("pop_rank") is not None and len(reasons) < 2:
            reasons.append("Trending this week")
        return reasons[:2] or ["Selected by the ranking model"]

    def recommend(self, c: int, k: int = 12) -> dict[str, Any]:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str, start: float) -> float:
            now = time.perf_counter()
            timings[name] = round(1000 * (now - start), 1)
            return now

        cfg_b, n_cands = self.meta["baselines"], self.meta["candidates"]
        hist = self.history(c).lazy()
        cust = pl.DataFrame({"customer_idx": [c]}, schema={"customer_idx": pl.Int32})
        rc, cc = cfg_b["repurchase"], cfg_b["cooccurrence"]
        sources: dict[str, pl.DataFrame] = {
            "rep": repurchase_candidates(
                hist, self.week, cust, self.ref_day, rc["n_weeks"], rc["max_per_user"]
            )
        }
        t = lap("baselines_ms", t0)

        retr_feats = {}
        for name in self.retrievers:
            df, u = self._retrieve(name, c, n_cands[name])
            sources[name] = df
            retr_feats[name] = {
                "user_ids": np.array([c]),
                "user_emb": u,
                "item_emb": self.retrievers[name].item_emb,
            }
        sources["co"] = cooccurrence_candidates(
            hist, self.week, cust, self.cooc, self.ref_day, cc["source_n_weeks"],
            cc["source_items"],
        )
        sources["pop"] = popularity_candidates(cust, self.popular)
        t = lap("retrieval_ms", t)

        cands = generate_candidates(sources, n_cands)
        # Tables globales reduites aux seuls candidats / au seul client : jointures en ~ms
        in_cands = pl.col("article_idx").is_in(cands["article_idx"].unique().to_list())
        feats = build_features(
            cands, hist, self.week, self.ref_day,
            self.articles.filter(in_cands),
            self.item_cat_df.filter(in_cands),
            self.user_attr_df.slice(c, 1),
            retr_feats,
            item_stats_df=self.item_stats.filter(in_cands),
            clip_emb=self.clip,
        )
        t = lap("features_ms", t)

        scores = self.booster.predict(to_matrix(feats, self.features))
        top = (
            feats.with_columns(pl.Series("score", scores.astype(np.float32)))
            .sort("score", descending=True)
            .head(k)
            .join(self.article_info, on="article_idx", how="left")
            .sort("score", descending=True)
        )
        recs = [
            {**self._article(row), "rank": i + 1, "score": round(float(row["score"]), 4),
             "reasons": self._reasons(row)}
            for i, row in enumerate(top.iter_rows(named=True))
        ]
        lap("ranking_ms", t)
        timings["total_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return {"n_candidates": cands.height, "latency_ms": timings, "recommendations": recs}

    def similar(self, a: int, k: int = 12) -> list[dict[str, Any]]:
        r = next(iter(self.retrievers.values()))
        _, pos = search(r.index, r.item_emb[a], k + 1)
        ids = [int(r.allowed[p]) for p in pos[0] if p >= 0 and int(r.allowed[p]) != a]
        return [self.article(i) for i in ids[:k]]
