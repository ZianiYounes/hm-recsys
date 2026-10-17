"""Index FAISS HNSW pour la recherche approchee des plus proches voisins (serving)."""
from __future__ import annotations

import faiss
import numpy as np


def build_hnsw_index(
    emb: np.ndarray, m: int = 32, ef_construction: int = 200, ef_search: int = 128
) -> faiss.Index:
    """Graphe HNSW en produit scalaire (vecteurs deja normalises -> similarite cosinus)."""
    vectors = np.ascontiguousarray(emb, dtype=np.float32)
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    index.hnsw.efSearch = ef_search
    return index


def search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (scores [Q, k], positions [Q, k]) ; une position -1 signifie "pas de resultat"."""
    q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
    return index.search(q, min(k, index.ntotal))
