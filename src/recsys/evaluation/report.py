"""Affichage console des resultats (tableau rich)."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

COLUMNS = [
    ("MAP@12", "map_at_12"),
    ("NDCG@12", "ndcg_at_12"),
    ("Recall@12", "recall_at_12"),
    ("Recall@50", "recall_at_50"),
    ("Recall@100", "recall_at_100"),
    ("Hit@12", "hit_rate_at_12"),
    ("Users %", "user_coverage"),
    ("Catalog@12 %", "catalog_coverage_at_12"),
]


def print_results_table(results: dict[str, dict[str, float]], title: str) -> None:
    table = Table(title=title)
    table.add_column("Modele", style="bold")
    for label, _ in COLUMNS:
        table.add_column(label, justify="right")
    for name, m in results.items():
        cells = []
        for _, key in COLUMNS:
            v = m.get(key, float("nan"))
            cells.append(f"{100 * v:.2f}" if "coverage" in key else f"{v:.4f}")
        table.add_row(name, *cells)
    Console().print(table)
