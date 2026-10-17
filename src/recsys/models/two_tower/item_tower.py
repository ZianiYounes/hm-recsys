from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ItemTower(nn.Module):
    """Article -> vecteur unitaire. Combine :
    - l'identifiant (memorisation des articles bien vendus) ;
    - les attributs (type, couleur, rayon...) qui generalisent aux articles peu vendus ;
    - en option, un embedding multimodal fige (FashionCLIP image + texte) qui permet de
      recommander un article neuf sur la seule base de son apparence (cold-start)."""

    def __init__(
        self,
        n_items: int,
        item_cats: torch.Tensor,   # [n_items, n_cats] codes entiers
        cat_cards: list[int],
        cat_dims: list[int],
        id_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        mm_emb: torch.Tensor | None = None,  # [n_items, M] embeddings FashionCLIP
        mm_dim: int = 64,
    ):
        super().__init__()
        self.register_buffer("item_cats", item_cats.long(), persistent=False)
        self.id_emb = nn.Embedding(n_items, id_dim)
        self.cat_embs = nn.ModuleList(
            nn.Embedding(card, dim) for card, dim in zip(cat_cards, cat_dims, strict=True)
        )
        in_dim = id_dim + sum(cat_dims)
        self.mm_proj: nn.Module | None = None
        if mm_emb is not None:
            self.register_buffer("mm_emb", mm_emb.float(), persistent=False)
            self.mm_proj = nn.Linear(mm_emb.shape[1], mm_dim)
            in_dim += mm_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.id_emb.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cats = self.item_cats[ids]
        parts = [self.id_emb(ids)] + [emb(cats[..., i]) for i, emb in enumerate(self.cat_embs)]
        if self.mm_proj is not None:
            parts.append(self.mm_proj(self.mm_emb[ids]))
        x = torch.cat([p.to(parts[0].dtype) for p in parts], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)
