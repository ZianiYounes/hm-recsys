from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from recsys.data.datasets import N_GAP_BUCKETS
from recsys.models.sequential.sasrec import SASRecEncoder


class AttentionPooling(nn.Module):
    """Moyenne des achats ponderee par une attention apprise."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(h).squeeze(-1).float().masked_fill(~mask, -1e4)
        w = torch.softmax(scores, dim=-1) * mask
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # historique vide -> vecteur nul
        return (w.unsqueeze(-1).to(h.dtype) * h).sum(dim=1)


class UserTower(nn.Module):
    """Client -> vecteur unitaire, a partir de :
    - ses derniers achats (encodes par la tour article) + leur anciennete, resumes par
      une attention ("attention") ou un transformer sequentiel ("transformer", SASRec) ;
    - ses features (age, statut club, newsletter...), utiles pour les clients froids."""

    def __init__(
        self,
        item_dim: int,
        user_cards: list[int],
        user_feat_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        encoder: str = "attention",
        max_len: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.gap_emb = nn.Embedding(N_GAP_BUCKETS, item_dim, padding_idx=0)
        if encoder == "transformer":
            self.seq: nn.Module = SASRecEncoder(item_dim, max_len, n_layers, n_heads, dropout)
        elif encoder == "attention":
            self.seq = AttentionPooling(item_dim)
        else:
            raise ValueError(f"encoder inconnu : {encoder}")
        self.feat_embs = nn.ModuleList(nn.Embedding(c, user_feat_dim) for c in user_cards)
        in_dim = item_dim + 1 + user_feat_dim * len(user_cards)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        hist_emb: torch.Tensor,    # [B, L, D] embeddings des achats passes
        gap_bucket: torch.Tensor,  # [B, L]
        mask: torch.Tensor,        # [B, L] True = achat reel
        user_feats: torch.Tensor,  # [B, F]
    ) -> torch.Tensor:
        h = hist_emb + self.gap_emb(gap_bucket)
        pooled = self.seq(h, mask)
        n_hist = mask.sum(dim=1, keepdim=True).float().log1p().to(pooled.dtype)
        feats = [emb(user_feats[:, i]).to(pooled.dtype) for i, emb in enumerate(self.feat_embs)]
        x = torch.cat([pooled, n_hist, *feats], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)
