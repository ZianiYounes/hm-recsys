"""Metriques de recommandation (pertinence binaire, predictions sans doublon).

Toutes les fonctions prennent :
  preds   : liste ordonnee d'articles recommandes pour un client
  target  : articles reellement achetes pendant la semaine cible
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def ap_at_k(preds: Sequence[int], target: Sequence[int], k: int = 12) -> float:
    """Average Precision@K, definition de la competition H&M."""
    if not target:
        return 0.0
    tset = set(target)
    hits, score = 0, 0.0
    for i, p in enumerate(preds[:k]):
        if p in tset:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(tset), k)


def recall_at_k(preds: Sequence[int], target: Sequence[int], k: int) -> float:
    if not target:
        return 0.0
    tset = set(target)
    return len(tset.intersection(preds[:k])) / len(tset)


def hit_rate_at_k(preds: Sequence[int], target: Sequence[int], k: int) -> float:
    return float(bool(set(target).intersection(preds[:k])))


def ndcg_at_k(preds: Sequence[int], target: Sequence[int], k: int) -> float:
    if not target:
        return 0.0
    tset = set(target)
    dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(preds[:k]) if p in tset)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(tset), k)))
    return dcg / idcg


def compute_metrics(
    preds: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    ks: Sequence[int] = (12,),
    map_k: int = 12,
) -> dict[str, float]:
    """Moyennes sur tous les clients cibles. Cles compatibles MLflow (pas de '@')."""
    pairs = list(zip(preds, targets, strict=True))

    def mean(fn, k: int) -> float:
        return float(np.mean([fn(p, t, k) for p, t in pairs])) if pairs else 0.0

    out = {f"map_at_{map_k}": mean(ap_at_k, map_k)}
    for k in ks:
        out[f"ndcg_at_{k}"] = mean(ndcg_at_k, k)
        out[f"recall_at_{k}"] = mean(recall_at_k, k)
        out[f"hit_rate_at_{k}"] = mean(hit_rate_at_k, k)
    return out
