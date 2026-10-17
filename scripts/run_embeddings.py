"""python scripts/run_embeddings.py : embeddings FashionCLIP (image + texte) des articles."""
from recsys.config import PATHS, load_config
from recsys.embeddings.build_item_embeddings import build_item_embeddings
from recsys.utils.logging import setup_logging


def main() -> None:
    setup_logging(PATHS.logs / "embeddings.log")
    build_item_embeddings(PATHS, load_config("embeddings"))


if __name__ == "__main__":
    main()
