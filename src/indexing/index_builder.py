from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.indexing.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def _load_encoder(encoder_type: str, model_name: str, vector_dim: int):
    """Factory: return the appropriate encoder instance."""
    if encoder_type == "colpali":
        from src.models.colpali_encoder import ColPaliEncoder

        enc = ColPaliEncoder(model_name=model_name)
        return enc, enc.vector_dim
    else:
        from src.models.colpali_encoder import BaselineEncoder

        enc = BaselineEncoder(vector_dim=vector_dim)
        return enc, vector_dim


class IndexBuilder:
    def __init__(
        self,
        manifest_path: Path,
        qdrant_path: Path,
        collection_name: str = "vidore_pages",
        vector_dim: int = 128,
        batch_size: int = 16,
        encoder_type: str = "colpali",
        model_name: str = "vidore/colqwen2-v1.0",
    ) -> None:
        self.manifest_path = manifest_path
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.encoder_type = encoder_type

        self.encoder, resolved_dim = _load_encoder(encoder_type, model_name, vector_dim)
        self.vector_dim = resolved_dim

        self.store = QdrantVectorStore(
            path=str(qdrant_path),
            collection_name=collection_name,
            vector_size=resolved_dim,
        )
        self.progress_path = manifest_path.parent / "index_progress.json"
        # Directory to cache per-page multi-vectors for later MaxSim reranking
        self.mv_cache_dir = manifest_path.parent / "mv_cache"
        self.mv_cache_dir.mkdir(parents=True, exist_ok=True)

    def _read_manifest(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load_progress(self) -> int:
        if not self.progress_path.exists():
            return 0
        with self.progress_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("next_index", 0))

    def _save_progress(self, next_index: int) -> None:
        with self.progress_path.open("w", encoding="utf-8") as f:
            json.dump({"next_index": next_index}, f, ensure_ascii=False, indent=2)

    def build(self, limit: int | None = None) -> dict[str, int]:
        rows = self._read_manifest()
        start = self._load_progress()
        end = len(rows) if limit is None else min(len(rows), start + limit)

        batch_rows: list[tuple[int, dict[str, Any]]] = []
        batch_images: list[Image.Image] = []
        indexed = 0
        skipped = 0
        point_id = start

        pbar = tqdm(range(start, end), desc="Indexing pages", unit="page")
        for i in pbar:
            row = rows[i]
            image_path = row.get("image_path")
            if not image_path or not Path(image_path).exists():
                skipped += 1
                continue

            batch_rows.append((point_id, row))
            if self.encoder_type == "colpali":
                batch_images.append(Image.open(image_path).convert("RGB"))
            point_id += 1

            if len(batch_rows) >= self.batch_size:
                indexed += self._flush_batch(batch_rows, batch_images)
                batch_rows = []
                batch_images = []
                self._save_progress(i + 1)

        if batch_rows:
            indexed += self._flush_batch(batch_rows, batch_images)

        self._save_progress(end)
        return {
            "indexed": indexed,
            "skipped": skipped,
            "processed_until": end,
            "total_manifest_rows": len(rows),
            "collection_count": self.store.count(),
        }

    def _flush_batch(
        self,
        batch_rows: list[tuple[int, dict[str, Any]]],
        batch_images: list[Image.Image],
    ) -> int:
        """Encode + upsert one batch of pages."""
        points = []

        if self.encoder_type == "colpali":
            # Batch encode images → list of multi-vector arrays
            mv_list = self.encoder.encode_images_batch(batch_images)
            for (pid, row), mv in zip(batch_rows, mv_list):
                # Mean-pool for Qdrant single-vector storage
                pooled = mv.mean(axis=0)
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    pooled = pooled / norm

                # Cache raw multi-vector for later MaxSim reranking
                cache_path = self.mv_cache_dir / f"{pid}.npy"
                np.save(str(cache_path), mv)

                payload = self._build_payload(row, pid)
                points.append(
                    self.store.build_point(
                        point_id=pid,
                        vector=pooled.astype(np.float32).tolist(),
                        payload=payload,
                    )
                )
            # Close batch images
            for img in batch_images:
                img.close()
        else:
            for pid, row in batch_rows:
                vector = self.encoder.encode_image(row["image_path"]).tolist()
                payload = self._build_payload(row, pid)
                points.append(
                    self.store.build_point(point_id=pid, vector=vector, payload=payload)
                )

        self.store.upsert(points)
        return len(points)

    @staticmethod
    def _build_payload(row: dict[str, Any], pid: int) -> dict[str, Any]:
        return {
            "point_id": pid,
            "split": row.get("split"),
            "query_id": row.get("query_id"),
            "query_text": row.get("query_text"),
            "doc_id": row.get("doc_id"),
            "page_id": row.get("page_id"),
            "page_num": row.get("page_num"),
            "language": row.get("language"),
            "image_path": row.get("image_path"),
        }
