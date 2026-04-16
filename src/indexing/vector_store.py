from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def upsert(self, points: list[PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count

    @staticmethod
    def build_point(point_id: int, vector: list[float], payload: dict[str, Any]) -> PointStruct:
        return PointStruct(id=point_id, vector=vector, payload=payload)

    # ------------------------------------------------------------------
    # Read / search
    # ------------------------------------------------------------------
    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[ScoredPoint]:
        """
        Cosine-similarity search against the mean-pooled page vectors.
        Returns top-k scored points with payloads.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def get_vectors_by_ids(self, point_ids: list[int]) -> dict[int, list[float]]:
        """
        Retrieve raw vectors for the given point IDs (needed for MaxSim reranking).
        """
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_vectors=True,
            with_payload=False,
        )
        return {p.id: p.vector for p in points}
