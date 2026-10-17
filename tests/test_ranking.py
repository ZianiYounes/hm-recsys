import numpy as np
import polars as pl

from recsys.features.feature_store import two_tower_similarity
from recsys.models.ranking import add_labels, generate_candidates


def _src(rows):
    return pl.DataFrame(rows, schema=["customer_idx", "article_idx", "score"], orient="row")


def test_candidates_union_ranks_and_sources():
    rep = _src([(0, 1, 5.0), (0, 2, 1.0)])
    tt = _src([(0, 2, 0.9), (0, 3, 0.8), (0, 4, 0.1)])
    out = generate_candidates({"rep": rep, "tt": tt}, {"rep": 10, "tt": 2})
    assert out["article_idx"].to_list() == [1, 2, 3]       # article 4 coupe (tt top 2)
    row2 = out.filter(pl.col("article_idx") == 2).row(0, named=True)
    assert (row2["rep_rank"], row2["tt_rank"], row2["n_sources"]) == (2, 1, 2)
    assert out.filter(pl.col("article_idx") == 3)["rep_rank"].is_null().all()


def test_labels():
    cands = pl.DataFrame({"customer_idx": [0, 0, 1], "article_idx": [1, 2, 1]},
                         schema={"customer_idx": pl.Int32, "article_idx": pl.Int32})
    targets = pl.DataFrame({"customer_idx": [0], "purchased": [[2, 9]]})
    out = add_labels(cands, targets).sort(["customer_idx", "article_idx"])
    assert out["label"].to_list() == [0, 1, 0]


def test_two_tower_similarity_unknown_user_is_nan():
    tt = {"user_ids": np.array([5]), "user_emb": np.array([[1.0, 0.0]], dtype=np.float32),
          "item_emb": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    sim = two_tower_similarity(np.array([5, 5, 7]), np.array([0, 1, 0]), tt)
    assert sim[0] == 1.0 and sim[1] == 0.0 and np.isnan(sim[2])
