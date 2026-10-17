"""LightGBM LambdaRank : apprend a ordonner les candidats de chaque client."""
from __future__ import annotations

import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from recsys.evaluation.evaluate import candidates_to_lists

GROUP_KEY = "group_key"


def to_matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in features]).to_numpy()


def group_sizes(df: pl.DataFrame) -> np.ndarray:
    """Tailles des groupes consecutifs (un groupe = un client d'une semaine)."""
    return df.group_by(GROUP_KEY, maintain_order=True).len()["len"].to_numpy()


def train_lgbm_ranker(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame | None,
    features: list[str],
    categorical: list[str],
    params: dict[str, Any],
    log_every: int = 50,
) -> lgb.LGBMRanker:
    params = dict(params)
    eval_at = params.pop("eval_at", [12])
    model = lgb.LGBMRanker(**params)
    with warnings.catch_warnings():  # eval_set renomme dans les versions recentes de LightGBM
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        _fit(model, train_df, valid_df, features, categorical, eval_at, log_every)
    return model


def _fit(model, train_df, valid_df, features, categorical, eval_at, log_every) -> None:
    eval_kwargs = {}
    if valid_df is not None:  # suivi de la courbe uniquement, pas d'early stopping
        eval_kwargs = {
            "eval_set": [(to_matrix(valid_df, features), valid_df["label"].to_numpy())],
            "eval_group": [group_sizes(valid_df)],
            "eval_at": eval_at,
            "callbacks": [lgb.log_evaluation(log_every)],
        }
    model.fit(
        to_matrix(train_df, features),
        train_df["label"].to_numpy(),
        group=group_sizes(train_df),
        feature_name=features,
        categorical_feature=[c for c in categorical if c in features],
        **eval_kwargs,
    )


def rank_candidates(
    model: lgb.LGBMRanker, df: pl.DataFrame, features: list[str], k: int
) -> pl.DataFrame:
    """-> (customer_idx, preds) : top-k articles par score du ranker."""
    scores = model.booster_.predict(to_matrix(df, features))
    scored = df.select("customer_idx", "article_idx").with_columns(
        pl.Series("score", scores.astype(np.float32))
    )
    return candidates_to_lists(scored, k)


def feature_importance(model: lgb.LGBMRanker, features: list[str]) -> pl.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    return (
        pl.DataFrame({"feature": features, "gain": gain.astype(np.float64)})
        .with_columns((100 * pl.col("gain") / pl.col("gain").sum()).alias("gain_pct"))
        .sort("gain", descending=True)
    )
