from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class ColPaliEncoder:
    """
    Lightweight image encoder shim for indexing.

    This intentionally keeps the interface stable for later drop-in
    replacement with a full ColPali/ColQwen implementation.
    """

    def __init__(self, vector_dim: int = 256) -> None:
        self.vector_dim = vector_dim

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)

        # Fast deterministic baseline embedding from image statistics.
        # Sprint follow-up: replace with true ColPali multi-vector embeddings.
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        h, w = arr.shape[:2]
        aspect = np.array([w / max(h, 1)], dtype=np.float32)
        stats = np.concatenate([mean_rgb, std_rgb, aspect], axis=0)

        # Tile and trim to fixed dimensionality.
        tiled = np.tile(stats, int(np.ceil(self.vector_dim / len(stats))))
        vector = tiled[: self.vector_dim]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)
