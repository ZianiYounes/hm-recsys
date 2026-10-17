"""python scripts/run_baselines.py [--split val|test]

Calcule popularite, re-achat, co-achat et leur fusion, puis les evalue sur le split.
Resultats : console, artifacts/reports/baselines_<split>.json, MLflow (experiment "baselines")."""
import argparse

import polars as pl

from recsys.config import PATHS, load_config
from recsys.data.split import load_split, target_week_of
from recsys.evaluation.evaluate import candidates_to_lists, evaluate_predictions
from recsys.evaluation.report import print_results_table
from recsys.models.baselines import baseline_candidates, blend_by_priority
from recsys.utils.io import read_json, write_json
from recsys.utils.logging import setup_logging
from recsys.utils.timer import timer
from recsys.utils.tracking import log_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    setup_logging(PATHS.logs / "baselines.log")
    cfg = load_config("baselines")
    ks, max_k = cfg["ks"], max(cfg["ks"])
    n_articles = read_json(PATHS.processed / "meta.json")["n_articles"]

    week = target_week_of(PATHS, args.split)
    history_lazy, targets = load_split(PATHS, args.split)
    with timer("chargement historique"):
        history = history_lazy.collect().lazy()  # une seule lecture disque
    ref_day = history.select(pl.col("day").max()).collect().item()
    customers = targets.select("customer_idx")

    cands = baseline_candidates(history, week, customers, ref_day, cfg)

    results: dict[str, dict] = {}
    with timer("evaluation"):
        for name, c in cands.items():
            results[name] = evaluate_predictions(
                candidates_to_lists(c, max_k), targets, ks, n_articles
            )
        blended = blend_by_priority(
            [cands["re-achat"], cands["co-achat"], cands["popularite"]], k=max_k
        )
        results["fusion"] = evaluate_predictions(blended, targets, ks, n_articles)

    print_results_table(results, f"Baselines ({args.split})")
    write_json(results, PATHS.reports / f"baselines_{args.split}.json")
    for name, metrics in results.items():
        log_run("baselines", f"{name}_{args.split}", {"split": args.split, **cfg}, metrics)


if __name__ == "__main__":
    main()
