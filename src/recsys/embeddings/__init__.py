"""Embeddings multimodaux des articles (FashionCLIP image + texte), Phase 3."""
from __future__ import annotations

import numpy as np

from recsys.config import Paths

ITEM_EMB_FILE = "fashionclip_items.npy"


def load_item_embeddings(paths: Paths, required: bool = True) -> np.ndarray | None:
    path = paths.embeddings / ITEM_EMB_FILE
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{path} introuvable : lance d'abord `python scripts/run_embeddings.py`"
            )
        return None
    return np.load(path).astype(np.float32)
