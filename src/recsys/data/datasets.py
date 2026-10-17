"""Historiques d'achats au format CSR + echantillons d'entrainement, assembles sur GPU.

Pas de DataLoader : tout l'historique (31M achats) tient sur la RTX 5080 (~250 Mo),
et chaque batch est construit par indexation vectorisee, sans boucle Python.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

N_GAP_BUCKETS = 12  # 0 = padding, puis log2(jours ecoules) + 1


@dataclass
class SequenceStore:
    items: np.ndarray    # [T] article_idx, trie par (client, jour)
    days: np.ndarray     # [T]
    weeks: np.ndarray    # [T]
    offsets: np.ndarray  # [n_customers + 1] : achats du client c = [offsets[c], offsets[c+1])

    @classmethod
    def from_history(cls, history: pl.DataFrame, n_customers: int) -> SequenceStore:
        h = (
            history.select("customer_idx", "day", "week", "article_idx")
            .unique(["customer_idx", "day", "article_idx"])
            .sort(["customer_idx", "day", "article_idx"])
        )
        counts = np.bincount(h["customer_idx"].to_numpy(), minlength=n_customers)
        offsets = np.zeros(n_customers + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return cls(
            items=h["article_idx"].to_numpy().astype(np.int64),
            days=h["day"].to_numpy().astype(np.int64),
            weeks=h["week"].to_numpy().astype(np.int64),
            offsets=offsets,
        )

    def customer_of_position(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.offsets) - 1), np.diff(self.offsets))

    def training_samples(self, target_week: int, train_weeks: int) -> dict[str, np.ndarray]:
        """Un exemple par achat des `train_weeks` semaines precedant la semaine cible.

        Contexte d'un exemple = achats du client strictement avant le jour de l'achat
        (les achats du meme jour ne sont pas visibles : pas de fuite)."""
        cust = self.customer_of_position()
        n = len(self.items)
        new_group = np.ones(n, dtype=bool)
        new_group[1:] = (cust[1:] != cust[:-1]) | (self.days[1:] != self.days[:-1])
        group_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))

        keep = self.weeks <= target_week + train_weeks
        return {
            "customer": cust[keep],
            "item": self.items[keep],
            "hist_start": self.offsets[cust[keep]],
            "hist_end": group_start[keep],
            "target_day": self.days[keep],
        }


class GpuHistory:
    """Copie GPU des historiques + extraction des L derniers achats pour un batch."""

    def __init__(self, store: SequenceStore, device: torch.device):
        self.items = torch.from_numpy(store.items.astype(np.int32)).to(device)
        self.days = torch.from_numpy(store.days.astype(np.int32)).to(device)
        self.offsets = torch.from_numpy(store.offsets).to(device)
        self.device = device

    def gather(
        self,
        hist_start: torch.Tensor,
        hist_end: torch.Tensor,
        target_day: torch.Tensor,
        max_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (articles [B, L], tranche de recence [B, L], masque [B, L]), du plus ancien au
        plus recent ; les positions vides valent 0 et sont masquees."""
        steps = torch.arange(max_len, 0, -1, device=self.device)
        idx = hist_end[:, None] - steps[None, :]
        mask = idx >= hist_start[:, None]
        idx = idx.clamp_min(0)
        if self.items.numel() == 0:
            zeros = torch.zeros_like(idx)
            return zeros, zeros, torch.zeros_like(mask)

        items = self.items[idx].long().masked_fill(~mask, 0)
        gap = (target_day[:, None] - self.days[idx]).clamp_min(1).float()
        bucket = (torch.log2(gap).floor().long() + 1).clamp(1, N_GAP_BUCKETS - 1)
        return items, bucket.masked_fill(~mask, 0), mask

    def last_purchases(
        self, customers: torch.Tensor, target_day: int, max_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Historique complet des clients (inference sur la semaine cible)."""
        start, end = self.offsets[customers], self.offsets[customers + 1]
        day = torch.full_like(customers, target_day)
        return self.gather(start, end, day, max_len)
