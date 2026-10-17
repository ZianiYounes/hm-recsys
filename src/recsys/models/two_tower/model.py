from __future__ import annotations

from typing import Any

import torch
from torch import nn

from recsys.features.item_features import ITEM_CAT_COLS
from recsys.models.two_tower.item_tower import ItemTower
from recsys.models.two_tower.user_tower import UserTower


class TwoTower(nn.Module):
    """Deux reseaux qui projettent clients et articles dans le meme espace :
    score(client, article) = produit scalaire -> recherche par plus proches voisins.

    cfg["user_encoder"] = "attention" (two-tower classique) ou "transformer" (SASRec)."""

    def __init__(
        self,
        n_items: int,
        item_cats: torch.Tensor,
        item_cards: list[int],
        user_cards: list[int],
        cfg: dict[str, Any],
        mm_emb: torch.Tensor | None = None,
        max_len: int = 32,
    ):
        super().__init__()
        cat_dims = [
            cfg["product_code_dim"] if c == "product_code" else cfg["cat_dim"]
            for c in ITEM_CAT_COLS
        ]
        self.item_tower = ItemTower(
            n_items=n_items,
            item_cats=item_cats,
            cat_cards=item_cards,
            cat_dims=cat_dims,
            id_dim=cfg["id_dim"],
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg["out_dim"],
            dropout=cfg["dropout"],
            mm_emb=mm_emb,
            mm_dim=cfg.get("mm_dim", 64),
        )
        self.user_tower = UserTower(
            item_dim=cfg["out_dim"],
            user_cards=user_cards,
            user_feat_dim=cfg["user_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            out_dim=cfg["out_dim"],
            dropout=cfg["dropout"],
            encoder=cfg.get("user_encoder", "attention"),
            max_len=max_len,
            n_layers=cfg.get("n_layers", 2),
            n_heads=cfg.get("n_heads", 4),
        )

    def encode_items(self, ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(ids)

    def encode_users(
        self,
        hist_items: torch.Tensor,
        gap_bucket: torch.Tensor,
        mask: torch.Tensor,
        user_feats: torch.Tensor,
    ) -> torch.Tensor:
        hist_emb = self.item_tower(hist_items)  # tour article partagee
        return self.user_tower(hist_emb, gap_bucket, mask, user_feats)
