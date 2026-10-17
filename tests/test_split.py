import datetime as dt

import polars as pl

from recsys.data.split import history_for, targets_for


def _tx() -> pl.DataFrame:
    end = dt.date(2020, 9, 22)
    rows = [
        # (jours avant la fin, client, article)
        (0, 0, 10), (1, 0, 11), (2, 0, 10),   # semaine 0, article 10 en double
        (8, 0, 12),                           # semaine 1
        (15, 1, 13), (9, 1, 14),              # semaines 2 et 1
        (30, 2, 15),                          # semaine 4
    ]
    return pl.DataFrame(
        {
            "t_dat": [end - dt.timedelta(days=d) for d, _, _ in rows],
            "week": [d // 7 for d, _, _ in rows],
            "customer_idx": [c for _, c, _ in rows],
            "article_idx": [a for _, _, a in rows],
        }
    )


def test_history_never_sees_target_week_or_future():
    h = history_for(_tx(), target_week=1).collect()
    assert h["week"].min() > 1


def test_history_window():
    h = history_for(_tx(), target_week=0, history_weeks=2).collect()
    assert set(h["week"].unique()) <= {1, 2}


def test_targets_unique_and_chronological():
    t = targets_for(_tx(), target_week=0).collect()
    row = t.filter(pl.col("customer_idx") == 0)
    # achats : 10 (j-2), 11 (j-1), 10 (j-0) -> [10, 11]
    assert row["purchased"].to_list() == [[10, 11]]


def test_targets_only_target_week_customers():
    t = targets_for(_tx(), target_week=1).collect()
    assert sorted(t["customer_idx"].to_list()) == [0, 1]
