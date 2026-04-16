from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.indexing.vector_store import QdrantVectorStore
from src.models.colpali_encoder import ColPaliEncoder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single scored result returned to the caller."""
    point_id: int
    score: float
    doc_id: str
    page_id: str
    page_num: int | None
    query_text: str | None
    image_path: str | None
    split: str | None
    language: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "score": round(self.score, 4),
            "doc_id": self.doc_id,
            "page_id": self.page_id,
            "page_num": self.page_num,
            "query_text": self.query_text,
            "image_path": self.image_path,
            "split": self.split,
            "language": self.language,
        }


class Retriever:
    """
    Two-stage retrieval pipeline:
        1. Candidate retrieval via mean-pooled cosine search in Qdrant.
        2. MaxSim reranking using cached multi-vector embeddings.
    """

    def __init__(
        self,
        encoder: ColPaliEncoder,
        store: QdrantVectorStore,
        mv_cache_dir: Path,
        candidate_pool: int = 50,
    ) -> None:
        self.encoder = encoder
        self.store = store
        self.mv_cache_dir = mv_cache_dir
        self.candidate_pool = candidate_pool

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Full retrieval pipeline: encode query → candidate search → MaxSim rerank → top-k.
        """
        # Stage 1: Encode query (mean-pooled) and find candidates via cosine
        query_pooled = self.encoder.encode_query_pooled(query_text)
        candidates = self.store.search(
            query_vector=query_pooled.tolist(),
            top_k=self.candidate_pool,
        )

        if not candidates:
            return []

        # Stage 2: MaxSim reranking with multi-vector embeddings
        query_mv = self.encoder.encode_query(query_text)

        scored: list[tuple[float, Any]] = []
        for c in candidates:
            pid = c.id
            payload = c.payload or {}

            # Load cached multi-vector for this page
            cache_path = self.mv_cache_dir / f"{pid}.npy"
            if cache_path.exists():
                doc_mv = np.load(str(cache_path))
                score = ColPaliEncoder.maxsim_score(query_mv, doc_mv)
            else:
                # Fallback: use the cosine score from stage 1
                score = float(c.score) * 100  # scale to be roughly comparable
                logger.warning("No cached MV for point %s — falling back to cosine score.", pid)

            scored.append((score, c))

        # Sort by MaxSim score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results = []
        for score, c in top:
            p = c.payload or {}
            results.append(
                RetrievalResult(
                    point_id=c.id,
                    score=score,
                    doc_id=p.get("doc_id", ""),
                    page_id=p.get("page_id", ""),
                    page_num=p.get("page_num"),
                    query_text=p.get("query_text"),
                    image_path=p.get("image_path"),
                    split=p.get("split"),
                    language=p.get("language"),
                )
            )
        return results
