"""Construction d'un TwoTower a partir de la config (entrainement et serving)."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch

from recsys.config import Paths
from recsys.features.item_features import encode_item_categoricals
from recsys.features.user_features import encode_user_features
from recsys.models.two_tower.model import TwoTower


def load_multimodal(paths: Paths, cfg: dict[str, Any]) -> np.ndarray | None:
    mc = cfg["model"]
    if not mc.get("multimodal", False):
        return None
    path = paths.embeddings / mc.get("mm_file", "fashionclip_items.npy")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable : lance d'abord `python scripts/run_embeddings.py` (Phase 3)"
        )
    return np.load(path).astype(np.float32)


def build_two_tower(
    paths: Paths, cfg: dict[str, Any], articles: pl.DataFrame, customers: pl.DataFrame
) -> tuple[TwoTower, np.ndarray]:
    """-> (modele non entraine, features clients encodees [n_customers, F])."""
    item_cats, item_cards = encode_item_categoricals(articles)
    user_feats, user_cards = encode_user_features(customers)
    mm = load_multimodal(paths, cfg)
    model = TwoTower(
        n_items=articles.height,
        item_cats=torch.from_numpy(item_cats),
        item_cards=item_cards,
        user_cards=user_cards,
        cfg=cfg["model"],
        mm_emb=None if mm is None else torch.from_numpy(mm),
        max_len=cfg["data"]["max_history"],
    )
    return model, user_feats
