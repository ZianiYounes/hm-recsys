"""Encodage des descriptions produits (nom, type, couleur, rayon, description) avec FashionCLIP."""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from recsys.embeddings.image_encoder import as_features


@torch.no_grad()
def encode_texts(
    model, processor, texts: list[str], device: torch.device, batch_size: int
) -> np.ndarray:
    out = []
    for s in tqdm(range(0, len(texts), batch_size), desc="textes"):
        tokens = processor(
            text=texts[s : s + batch_size], padding=True, truncation=True, max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats = as_features(
                model.get_text_features(
                    input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
                ),
                "text_embeds",
            )
        out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(out).numpy()
