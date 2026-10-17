"""python scripts/run_download.py            -> CSV uniquement (~3.5 Go)
python scripts/run_download.py --images   -> CSV + images (~30 Go, utile en Phase 3)"""
import argparse

from recsys.config import PATHS, load_config
from recsys.data.download import download_dataset
from recsys.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", action="store_true", help="telecharger aussi les images")
    args = parser.parse_args()

    setup_logging(PATHS.logs / "download.log")
    cfg = load_config("data")
    download_dataset(
        competition=cfg["competition"],
        raw_dir=PATHS.raw,
        files=cfg["raw_files"],
        with_images=args.images or cfg.get("download_images", False),
    )


if __name__ == "__main__":
    main()
