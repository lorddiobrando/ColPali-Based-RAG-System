from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ColPaliEncoder:
    """
    Production ColPali engine encoder for document page images and text queries.

    Uses the `colpali-engine` library to produce multi-vector embeddings.
    Supports vidore models like colqwen2-v1.0 (Qwen2) and colSmol-256M (Idefics3).

    Storage strategy (Option B):
        - Mean-pool patch vectors into a single vector per page for Qdrant search.
        - Keep raw multi-vectors in memory for MaxSim reranking on the top-k.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        logger.info("Loading Vision Model %s on %s …", model_name, device)
        
        # Dynamically load the correct architecture class
        if "smol" in model_name.lower() or "idefics" in model_name.lower():
            from colpali_engine.models import ColIdefics3 as Architecture
            from colpali_engine.models import ColIdefics3Processor as Processor
        else:
            from colpali_engine.models import ColQwen2 as Architecture
            from colpali_engine.models import ColQwen2Processor as Processor

        self.model = Architecture.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        ).eval()
        self.processor = Processor.from_pretrained(model_name)
        self._dim: int | None = None
        logger.info("%s model loaded.", model_name)

    @property
    def vector_dim(self) -> int:
        """Return the per-patch embedding dimensionality (typically 128)."""
        if self._dim is None:
            # Probe with a tiny dummy image to discover dim.
            dummy = Image.new("RGB", (32, 32), (128, 128, 128))
            mv = self.encode_image_multivector(dummy)
            self._dim = mv.shape[1]
        return self._dim

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------
    def encode_image_multivector(self, image: Image.Image) -> np.ndarray:
        """
        Encode one page image → multi-vector embeddings.
        Returns: np.ndarray of shape (num_patches, dim).
        """
        batch = self.processor.process_images([image])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            embeddings = self.model(**batch)  # (1, num_patches, dim)
        return embeddings[0].float().cpu().numpy()

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """
        Encode a page image file → mean-pooled single vector for Qdrant.
        Returns: np.ndarray of shape (dim,).
        """
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
        mv = self.encode_image_multivector(rgb)
        pooled = mv.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    def encode_images_batch(
        self, images: List[Image.Image]
    ) -> list[np.ndarray]:
        """
        Batch-encode multiple images → list of multi-vector arrays.
        """
        if not images:
            return []
        batch = self.processor.process_images(images)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            embeddings = self.model(**batch)  # (B, patches, dim)
        return [embeddings[i].float().cpu().numpy() for i in range(len(images))]

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------
    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode a text query → multi-vector embeddings.
        Returns: np.ndarray of shape (num_tokens, dim).
        """
        batch = self.processor.process_queries([query_text])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            embeddings = self.model(**batch)  # (1, tokens, dim)
        return embeddings[0].float().cpu().numpy()

    def encode_query_pooled(self, query_text: str) -> np.ndarray:
        """
        Encode query → mean-pooled single vector (for candidate retrieval).
        """
        mv = self.encode_query(query_text)
        pooled = mv.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    # ------------------------------------------------------------------
    # Late-interaction scoring (MaxSim)
    # ------------------------------------------------------------------
    @staticmethod
    def maxsim_score(
        query_mv: np.ndarray, doc_mv: np.ndarray
    ) -> float:
        """
        Compute the ColBERT-style MaxSim score between a multi-vector query
        and a multi-vector document.

        Score = Σ_i max_j (q_i · d_j)
        """
        # query_mv: (Q, D), doc_mv: (P, D) → sim: (Q, P)
        sim = query_mv @ doc_mv.T
        return float(sim.max(axis=1).sum())


class BaselineEncoder:
    """
    Fast deterministic baseline encoder from image statistics.
    Kept for comparison / CPU-only quick indexing.
    """

    def __init__(self, vector_dim: int = 128) -> None:
        self.vector_dim = vector_dim

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32)
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        h, w = arr.shape[:2]
        aspect = np.array([w / max(h, 1)], dtype=np.float32)
        stats = np.concatenate([mean_rgb, std_rgb, aspect], axis=0)
        tiled = np.tile(stats, int(np.ceil(self.vector_dim / len(stats))))
        vector = tiled[: self.vector_dim]
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)
