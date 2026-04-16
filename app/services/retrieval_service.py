from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.config.settings import AppSettings
from src.indexing.vector_store import QdrantVectorStore
from src.models.colpali_encoder import ColPaliEncoder
from src.retrieval.retrieve import Retriever, RetrievalResult

logger = logging.getLogger(__name__)

# Singleton-style lazy cache to avoid reloading model on every request.
_encoder_instance: ColPaliEncoder | None = None
_retriever_instance: Retriever | None = None


def _get_encoder(settings: AppSettings) -> ColPaliEncoder:
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = ColPaliEncoder(model_name=settings.colpali_model)
    return _encoder_instance


def _get_retriever(settings: AppSettings) -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        encoder = _get_encoder(settings)
        store = QdrantVectorStore(
            path=str(settings.qdrant_path),
            collection_name=settings.qdrant_collection,
            vector_size=encoder.vector_dim,
        )
        mv_cache_dir = settings.vidore_manifest_path.parent / "mv_cache"
        _retriever_instance = Retriever(
            encoder=encoder,
            store=store,
            mv_cache_dir=mv_cache_dir,
        )
    return _retriever_instance


def run_retrieval(
    settings: AppSettings, query: str, top_k: int = 5
) -> dict[str, Any]:
    """
    Execute full retrieval pipeline and return JSON-ready results with timing.
    """
    t0 = time.perf_counter()
    retriever = _get_retriever(settings)
    t_init = time.perf_counter() - t0

    t1 = time.perf_counter()
    results = retriever.retrieve(query, top_k=top_k)
    t_retrieval = time.perf_counter() - t1

    return {
        "query": query,
        "top_k": top_k,
        "results": [r.to_dict() for r in results],
        "timing": {
            "init_seconds": round(t_init, 3),
            "retrieval_seconds": round(t_retrieval, 3),
            "total_seconds": round(t_init + t_retrieval, 3),
        },
    }
