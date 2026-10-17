"""Etage de ranking : candidats (toutes sources) -> features -> LightGBM LambdaRank.

Pour evaluer sur la semaine W, le ranker est entraine sur les semaines W+1, W+2... :
il n'a jamais vu les achats de W, exactement comme en production."""
from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from rich.console import Console
from rich.table import Table

from recsys.config import Paths, load_config
from recsys.data.split import target_week_of
from recsys.embeddings import load_item_embeddings
from recsys.evaluation.evaluate import evaluate_predictions
from recsys.evaluation.report import print_results_table
from recsys.features.feature_store import CATEGORICAL_FEATURES, feature_columns
from recsys.features.item_features import item_categorical_frame
from recsys.features.user_features import user_attribute_frame
from recsys.models.baselines import blend_by_priority
from recsys.models.ranking import (
    build_week_dataset,
    feature_importance,
    rank_candidates,
    train_lgbm_ranker,
)
from recsys.models.ranking.ranker import GROUP_KEY
from recsys.training.train_two_tower import ensure_two_tower_cache
from recsys.utils.io import write_json, write_parquet
from recsys.utils.timer import timer
from recsys.utils.tracking import MlflowRun

KS = (12, 50, 100)


class RankerData:
    """Charge une fois les tables communes et construit les jeux de chaque semaine."""

    def __init__(
        self,
        paths: Paths,
        cfg: dict[str, Any],
        retriever_cfgs: dict[str, dict[str, Any]] | None = None,
    ):
        self.paths, self.cfg = paths, cfg
        self.retriever_cfgs = retriever_cfgs or {r: load_config(r) for r in cfg["retrievers"]}
        self.base_cfg = load_config("baselines")
        self.tx = pl.scan_parquet(paths.processed / "transactions.parquet")
        self.articles = pl.read_parquet(paths.processed / "articles.parquet")
        customers = pl.read_parquet(paths.processed / "customers.parquet")
        self.item_cat_df = item_categorical_frame(self.articles)
        self.user_attr_df = user_attribute_frame(customers)
        self.use_clip = bool(cfg.get("use_clip_features", False))
        self.clip_emb = load_item_embeddings(paths) if self.use_clip else None
        self.features = feature_columns(list(self.retriever_cfgs), self.use_clip)

    def week(self, week: int):
        retrievers = {
            name: ensure_two_tower_cache(self.paths, week, rcfg)
            for name, rcfg in self.retriever_cfgs.items()
        }
        return build_week_dataset(
            self.tx, week, retrievers, self.articles, self.item_cat_df, self.user_attr_df,
            self.cfg, self.base_cfg, self.clip_emb,
        )

    def training_table(self, weeks: list[int]) -> pl.DataFrame:
        parts = []
        for w in weeks:
            df, _, _ = self.week(w)
            # Les clients sans aucun achat parmi leurs candidats n'apprennent rien au ranker
            df = df.filter(pl.col("label").sum().over("customer_idx") > 0)
            parts.append(with_group(df, w))
        train_df = pl.concat(parts).sort([GROUP_KEY, "article_idx"])
        logger.info(
            f"Train ranker : {train_df.height:,} lignes, {train_df[GROUP_KEY].n_unique():,} "
            f"groupes, {100 * train_df['label'].mean():.2f} % de positifs"
        )
        return train_df


def with_group(df: pl.DataFrame, week: int) -> pl.DataFrame:
    return df.with_columns(
        (pl.lit(week, dtype=pl.Int64) * 10_000_000 + pl.col("customer_idx")).alias(GROUP_KEY)
    )


def print_importance(imp: pl.DataFrame, n: int = 15) -> None:
    table = Table(title=f"Top {n} features (gain)")
    table.add_column("Feature")
    table.add_column("Gain %", justify="right")
    for row in imp.head(n).iter_rows(named=True):
        table.add_row(row["feature"], f"{row['gain_pct']:.1f}")
    Console().print(table)


def run_ranker(
    paths: Paths,
    split: str,
    cfg: dict[str, Any],
    retriever_cfgs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    data = RankerData(paths, cfg, retriever_cfgs)
    eval_week = target_week_of(paths, split)
    train_weeks = [eval_week + d for d in cfg["train_week_offsets"]]
    logger.info(
        f"Ranker {list(data.retriever_cfgs)} : entrainement semaines {train_weeks}, "
        f"evaluation semaine {eval_week}"
    )

    train_df = data.training_table(train_weeks)
    eval_df, targets, sources = data.week(eval_week)
    eval_df = with_group(eval_df, eval_week).sort([GROUP_KEY, "article_idx"])
    n_target = int(targets["purchased"].list.len().sum())
    candidate_recall = float(eval_df["label"].sum()) / n_target
    n_items = data.articles.height

    run = MlflowRun("ranker", f"ranker_{split}", {"split": split, **cfg})
    with timer("entrainement LightGBM"):
        model = train_lgbm_ranker(
            train_df, eval_df, data.features, CATEGORICAL_FEATURES, cfg["lgbm"]
        )
    with timer("prediction"):
        preds = rank_candidates(model, eval_df, data.features, max(KS))

    retr = list(data.retriever_cfgs)
    blend_sources = [sources["rep"], *[sources[r] for r in retr], sources["co"], sources["pop"]]
    results = {
        f"fusion + {' + '.join(retr)}": evaluate_predictions(
            blend_by_priority(blend_sources, max(KS)), targets, KS, n_items
        ),
        "ranker LightGBM": evaluate_predictions(preds, targets, KS, n_items),
    }
    print_results_table(results, f"Ranker ({split})")
    logger.info(f"Plafond de rappel des candidats : {100 * candidate_recall:.1f} %")
    imp = feature_importance(model, data.features)
    print_importance(imp)

    paths.models.mkdir(parents=True, exist_ok=True)
    model_path = paths.models / f"ranker_{split}.txt"
    model.booster_.save_model(str(model_path))
    imp_path = paths.reports / f"ranker_{split}_importance.csv"
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    imp.write_csv(imp_path)
    write_parquet(preds, paths.interim / f"ranker_preds_{split}.parquet")
    write_json(
        {**results, "candidate_recall": candidate_recall, "retrievers": retr},
        paths.reports / f"ranker_{split}.json",
    )
    run.log({**results["ranker LightGBM"], "candidate_recall": candidate_recall})
    run.artifact(str(imp_path))
    run.end()
    logger.success(f"Ranker sauvegarde : {model_path}")
    return results


def train_final_ranker(
    paths: Paths,
    cfg: dict[str, Any],
    serve_week: int,
    out_dir,
    retriever_cfgs: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Ranker de production : entraine sur les semaines les plus recentes, sans evaluation."""
    data = RankerData(paths, cfg, retriever_cfgs)
    weeks = [serve_week + d for d in cfg["train_week_offsets"]]
    logger.info(f"Ranker de production : entrainement sur les semaines {weeks}")
    train_df = data.training_table(weeks)
    with timer("entrainement LightGBM"):
        model = train_lgbm_ranker(train_df, None, data.features, CATEGORICAL_FEATURES, cfg["lgbm"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_dir / "ranker.txt"))
    feature_importance(model, data.features).write_csv(out_dir / "ranker_importance.csv")
    write_json(data.features, out_dir / "ranker_features.json")
    return data.features
