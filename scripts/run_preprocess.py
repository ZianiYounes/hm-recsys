"""python scripts/run_preprocess.py : CSV bruts -> parquet dans data/processed/"""
from recsys.config import PATHS
from recsys.data.preprocess import run_preprocess
from recsys.utils.logging import setup_logging


def main() -> None:
    setup_logging(PATHS.logs / "preprocess.log")
    run_preprocess(PATHS)


if __name__ == "__main__":
    main()
