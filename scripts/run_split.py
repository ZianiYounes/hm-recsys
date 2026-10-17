"""python scripts/run_split.py : verites terrain val / test dans data/processed/splits/"""
from recsys.config import PATHS, load_config
from recsys.data.split import build_and_save_splits
from recsys.utils.logging import setup_logging


def main() -> None:
    setup_logging(PATHS.logs / "split.log")
    cfg = load_config("data")["split"]
    build_and_save_splits(
        PATHS,
        val_week=cfg["val_week"],
        test_week=cfg["test_week"],
        history_weeks=cfg.get("history_weeks"),
    )


if __name__ == "__main__":
    main()
