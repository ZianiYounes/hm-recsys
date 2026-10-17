"""Chemins du projet et chargement des configs YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(os.getenv("RECSYS_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    # ---- donnees ----
    @property
    def data(self) -> Path:
        return self.root / os.getenv("DATA_DIR", "data")

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def images(self) -> Path:
        return self.raw / "images"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def splits(self) -> Path:
        return self.processed / "splits"

    @property
    def features(self) -> Path:
        return self.data / "features"

    @property
    def embeddings(self) -> Path:
        return self.data / "embeddings"

    # ---- artefacts ----
    @property
    def artifacts(self) -> Path:
        return self.root / os.getenv("ARTIFACTS_DIR", "artifacts")

    @property
    def models(self) -> Path:
        return self.artifacts / "models"

    @property
    def indexes(self) -> Path:
        return self.artifacts / "indexes"

    @property
    def reports(self) -> Path:
        return self.artifacts / "reports"

    # ---- divers ----
    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


PATHS = Paths()


def load_config(name: str) -> dict[str, Any]:
    """Charge configs/<name>.yaml."""
    path = PATHS.configs / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
