"""Recherche exacte des top-K par produit scalaire sur GPU (evaluation offline).

Pour 72k clients x 105k articles, une multiplication matricielle par blocs prend quelques
secondes sur la RTX 5080 ; FAISS servira au serving temps reel (Phase 4)."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def topk_inner_product(
    queries: torch.Tensor,           # [U, D]
    items: torch.Tensor,             # [N, D]
    k: int,
    allowed: torch.Tensor | None = None,  # [N] bool : articles autorises
    chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (scores [U, k] float32, indices [U, k] int64), tries par score decroissant."""
    items_t = items.float().T
    k = min(k, items.shape[0] if allowed is None else int(allowed.sum()))
    all_s, all_i = [], []
    for start in range(0, queries.shape[0], chunk):
        scores = queries[start : start + chunk].float() @ items_t
        if allowed is not None:
            scores.masked_fill_(~allowed[None, :], float("-inf"))
        s, i = scores.topk(k, dim=1)
        all_s.append(s.cpu())
        all_i.append(i.cpu())
    return torch.cat(all_s).numpy(), torch.cat(all_i).numpy()
