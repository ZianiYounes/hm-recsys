"""python scripts/run_train.py --model two_tower|sasrec|ranker [--split val|test] [--epochs N]"""
import argparse

from recsys.config import PATHS, load_config
from recsys.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="two_tower", choices=["two_tower", "sasrec", "ranker"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--epochs", type=int, default=None, help="retrievers : surcharge la config")
    args = parser.parse_args()

    setup_logging(PATHS.logs / f"train_{args.model}.log")
    cfg = load_config(args.model)

    if args.model in ("two_tower", "sasrec"):
        from recsys.training.train_two_tower import run_two_tower

        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        run_two_tower(PATHS, args.split, cfg)
    else:
        from recsys.training.train_ranker import run_ranker

        run_ranker(PATHS, args.split, cfg)


if __name__ == "__main__":
    main()
