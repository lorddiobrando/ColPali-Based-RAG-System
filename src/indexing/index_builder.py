from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.indexing.vector_store import QdrantVectorStore
from src.models.colpali_encoder import ColPaliEncoder


class IndexBuilder:
    def __init__(
        self,
        manifest_path: Path,
        qdrant_path: Path,
        collection_name: str = "vidore_pages",
        vector_dim: int = 256,
        batch_size: int = 64,
    ) -> None:
        self.manifest_path = manifest_path
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.encoder = ColPaliEncoder(vector_dim=vector_dim)
        self.store = QdrantVectorStore(
            path=str(qdrant_path), collection_name=collection_name, vector_size=vector_dim
        )
        self.progress_path = manifest_path.parent / "index_progress.json"

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

        batch = []
        indexed = 0
        skipped = 0
        point_id = start

        for i in range(start, end):
            row = rows[i]
            image_path = row.get("image_path")
            if not image_path:
                skipped += 1
                continue
            if not Path(image_path).exists():
                skipped += 1
                continue

            vector = self.encoder.encode_image(image_path).tolist()
            payload = {
                "split": row.get("split"),
                "query_id": row.get("query_id"),
                "query_text": row.get("query_text"),
                "doc_id": row.get("doc_id"),
                "page_id": row.get("page_id"),
                "page_num": row.get("page_num"),
                "language": row.get("language"),
                "image_path": row.get("image_path"),
            }
            batch.append(self.store.build_point(point_id=point_id, vector=vector, payload=payload))
            point_id += 1
            indexed += 1

            if len(batch) >= self.batch_size:
                self.store.upsert(batch)
                batch = []
                self._save_progress(i + 1)

        if batch:
            self.store.upsert(batch)

        self._save_progress(end)
        return {
            "indexed": indexed,
            "skipped": skipped,
            "processed_until": end,
            "total_manifest_rows": len(rows),
            "collection_count": self.store.count(),
        }
