import math

import polars as pl
import pytest

from recsys.evaluation.metrics import ap_at_k, compute_metrics, ndcg_at_k, recall_at_k
from recsys.models.baselines import blend_by_priority


def test_ap_perfect_and_empty():
    assert ap_at_k([1, 2, 3], [1, 2, 3]) == 1.0
    assert ap_at_k([4, 5, 6], [1, 2]) == 0.0
    assert ap_at_k([1, 2], []) == 0.0


def test_ap_known_value():
    # hits en positions 1 et 3 -> (1/1 + 2/3) / min(2, 12)
    assert ap_at_k([1, 9, 2], [1, 2]) == pytest.approx((1 + 2 / 3) / 2)


def test_recall_and_ndcg():
    assert recall_at_k([1, 9, 2, 8], [1, 2, 3, 4], k=3) == 0.5
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([1, 9, 2], [1, 2], k=3) == pytest.approx(expected)


def test_compute_metrics_keys_are_mlflow_safe():
    m = compute_metrics([[1, 2]], [[2]], ks=(12, 50))
    assert all("@" not in k for k in m)
    assert {"map_at_12", "recall_at_50", "ndcg_at_12"} <= m.keys()


def test_blend_priority_and_dedup():
    rep = pl.DataFrame({"customer_idx": [0], "article_idx": [5], "score": [0.1]})
    pop = pl.DataFrame({"customer_idx": [0, 0], "article_idx": [7, 5], "score": [100.0, 90.0]})
    out = blend_by_priority([rep, pop], k=12)
    assert out["preds"].to_list() == [[5, 7]]
