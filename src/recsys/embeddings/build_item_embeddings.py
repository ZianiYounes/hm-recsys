"""Embeddings FashionCLIP de tous les articles -> data/embeddings/.

fashionclip_items.npy : [n_articles, 512] = moyenne normalisee image + texte
                        (texte seul pour les articles sans photo)"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger

from recsys.config import Paths
from recsys.embeddings import ITEM_EMB_FILE
from recsys.embeddings.image_encoder import encode_images
from recsys.embeddings.text_encoder import encode_texts
from recsys.utils.timer import timer


def build_item_embeddings(paths: Paths, cfg: dict[str, Any]) -> np.ndarray:
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(cfg["model"]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(cfg["model"])

    articles = pl.read_parquet(
        paths.processed / "articles.parquet", columns=["article_idx", "image_path", "text"]
    ).sort("article_idx")
    n = articles.height

    with timer("embeddings texte"):
        texts = [t or "" for t in articles["text"].to_list()]
        text_emb = encode_texts(model, processor, texts, device, cfg["batch_size"])

    img_paths = [paths.images / p for p in articles["image_path"].to_list()]
    has_image = np.array([p.exists() for p in img_paths])
    logger.info(f"Photos trouvees : {has_image.sum():,} / {n:,}")
    img_emb = np.zeros_like(text_emb)
    if has_image.any():
        with timer("embeddings images"):
            img_emb[has_image] = encode_images(
                model, processor, [p for p, ok in zip(img_paths, has_image, strict=True) if ok],
                device, cfg["batch_size"], cfg["num_workers"],
            )
    else:
        logger.warning("Aucune photo : lance `python scripts/run_download.py --images` d'abord")

    w = cfg["image_weight"]
    combined = np.where(has_image[:, None], w * img_emb + (1 - w) * text_emb, text_emb)
    combined /= np.linalg.norm(combined, axis=1, keepdims=True).clip(min=1e-6)

    out = paths.embeddings
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "fashionclip_text.npy", text_emb.astype(np.float16))
    np.save(out / "fashionclip_image.npy", img_emb.astype(np.float16))
    np.save(out / "has_image.npy", has_image)
    np.save(out / ITEM_EMB_FILE, combined.astype(np.float16))
    logger.success(f"Embeddings enregistres dans {out} ({combined.shape})")
    return combined
