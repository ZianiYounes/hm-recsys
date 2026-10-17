"""Encodeur sequentiel a base de transformer (style SASRec, Kang & McAuley 2018).

Lit les L derniers achats du client dans l'ordre et resume la sequence par la sortie
de la position la plus recente. Contrairement a la moyenne ponderee, il capte l'ordre
des achats (ex. : un jean puis une ceinture)."""
from __future__ import annotations

import torch
from torch import nn


class SASRecEncoder(nn.Module):
    def __init__(self, dim: int, max_len: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h : [B, L, D] (du plus ancien au plus recent), mask : [B, L] -> [B, D]."""
        length = h.shape[1]
        pos = torch.arange(length, device=h.device)
        x = h + self.pos_emb(pos)[None, :, :]

        # Un historique vide ferait un softmax sur rien (NaN) : on laisse passer une
        # position factice puis on remet la sortie a zero.
        empty = ~mask.any(dim=1)
        padding = ~mask
        padding[empty, -1] = False
        out = self.encoder(x, src_key_padding_mask=padding)
        last = self.norm(out[:, -1])
        return last.masked_fill(empty[:, None], 0.0)
