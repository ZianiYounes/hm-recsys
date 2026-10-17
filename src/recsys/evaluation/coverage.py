from __future__ import annotations

from collections.abc import Sequence


def catalog_coverage(preds: Sequence[Sequence[int]], n_articles: int, k: int = 12) -> float:
    """Part du catalogue qui apparait au moins une fois dans les top-K recommandes."""
    seen: set[int] = set()
    for p in preds:
        seen.update(p[:k])
    return len(seen) / n_articles if n_articles else 0.0


def user_coverage(preds: Sequence[Sequence[int]]) -> float:
    """Part des clients qui recoivent au moins une recommandation."""
    return sum(1 for p in preds if len(p) > 0) / max(len(preds), 1)
