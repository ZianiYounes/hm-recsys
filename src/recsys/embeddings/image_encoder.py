"""Encodage des photos produits avec FashionCLIP."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def as_features(out, attr: str) -> torch.Tensor:
    """Compatible avec les differentes versions de transformers (tenseur ou objet de sortie)."""
    if torch.is_tensor(out):
        return out
    for name in (attr, "pooler_output"):
        value = getattr(out, name, None)
        if value is not None:
            return value
    raise TypeError(f"Sortie inattendue du modele : {type(out)}")


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], processor):
        self.paths, self.processor = paths, processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as img:
            pixels = self.processor(images=img.convert("RGB"), return_tensors="pt")
        return pixels["pixel_values"][0]


@torch.no_grad()
def encode_images(
    model, processor, paths: list[Path], device: torch.device, batch_size: int, num_workers: int
) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(paths, processor),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    out = []
    for pixels in tqdm(loader, desc="images"):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats = as_features(model.get_image_features(pixel_values=pixels.to(device)),
                                "image_embeds")
        out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(out).numpy()
