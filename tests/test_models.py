import numpy as np
import polars as pl
import torch

from recsys.data.datasets import GpuHistory, SequenceStore
from recsys.models.two_tower import TwoTower, in_batch_softmax_loss

CFG = {"id_dim": 8, "cat_dim": 4, "product_code_dim": 4, "user_feat_dim": 2,
       "hidden_dim": 16, "out_dim": 8, "dropout": 0.0}


def _history() -> pl.DataFrame:
    # client 0 : jours 1, 1, 5, 9 ; client 2 : jour 3 ; client 1 : rien
    return pl.DataFrame({
        "customer_idx": [0, 0, 0, 0, 2],
        "day": [1, 1, 5, 9, 3],
        "week": [3, 3, 2, 1, 2],
        "article_idx": [10, 11, 12, 13, 14],
    })


def test_store_offsets_and_samples():
    store = SequenceStore.from_history(_history(), n_customers=3)
    assert store.offsets.tolist() == [0, 4, 4, 5]
    s = store.training_samples(target_week=0, train_weeks=1)  # achats de la semaine 1
    assert s["item"].tolist() == [13]
    assert (s["hist_start"][0], s["hist_end"][0]) == (0, 3)  # voit les 3 achats precedents


def test_same_day_purchases_not_visible():
    store = SequenceStore.from_history(_history(), n_customers=3)
    s = store.training_samples(target_week=0, train_weeks=5)
    # achats 10 et 11 (meme jour, premiers achats) : historique vide
    first_day = s["target_day"] == 1
    assert (s["hist_end"][first_day] == s["hist_start"][first_day]).all()


def test_gather_padding_and_order():
    store = SequenceStore.from_history(_history(), n_customers=3)
    hist = GpuHistory(store, torch.device("cpu"))
    items, gaps, mask = hist.last_purchases(torch.tensor([0, 1]), target_day=10, max_len=3)
    assert items[0].tolist() == [11, 12, 13]  # du plus ancien au plus recent
    assert mask[1].sum() == 0                 # client sans historique
    assert gaps[0, -1] == 1                   # achat de la veille -> tranche 1


def test_two_tower_shapes_and_loss():
    n_items = 20
    item_cats = torch.randint(0, 3, (n_items, 12))
    model = TwoTower(n_items, item_cats, [3] * 12, [4, 2, 2, 3, 3], CFG)
    b, length = 6, 5
    items = torch.randint(0, n_items, (b, length))
    mask = torch.ones(b, length, dtype=torch.bool)
    mask[0] = False  # client froid
    u = model.encode_users(items, torch.randint(0, 12, (b, length)), mask,
                           torch.zeros(b, 5, dtype=torch.long))
    pos = torch.tensor([1, 2, 3, 3, 4, 5])  # doublon volontaire
    v = model.encode_items(pos)
    assert u.shape == (b, 8) and torch.allclose(u.norm(dim=1), torch.ones(b), atol=1e-5)
    loss = in_batch_softmax_loss(u, v, pos, torch.zeros(n_items), 0.1)
    assert torch.isfinite(loss)
    loss.backward()


def test_topk_respects_allowed():
    from recsys.retrieval.torch_topk import topk_inner_product

    q, it = torch.eye(3), torch.eye(3)
    allowed = torch.tensor([True, False, True])
    _, idx = topk_inner_product(q, it, k=5, allowed=allowed)
    assert idx.shape == (3, 2) and 1 not in np.unique(idx)
