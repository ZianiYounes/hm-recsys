"""python scripts/build_serving.py : modeles et tables pour l'API (semaine apres les donnees)."""
from recsys.config import PATHS
from recsys.serving.build import build_serving
from recsys.utils.logging import setup_logging


def main() -> None:
    setup_logging(PATHS.logs / "build_serving.log")
    build_serving(PATHS)


if __name__ == "__main__":
    main()
