from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


class QdrantVectorStore:
    def __init__(self, path: str, collection_name: str, vector_size: int) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(path=path)
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, points: list[PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count

    @staticmethod
    def build_point(point_id: int, vector: list[float], payload: dict[str, Any]) -> PointStruct:
        return PointStruct(id=point_id, vector=vector, payload=payload)
