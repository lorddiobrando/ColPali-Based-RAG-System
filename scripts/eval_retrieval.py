"""
eval_retrieval.py
-----------------
Retrieval quality evaluation for the ColPali RAG system.

Metrics computed (per dataset and aggregate):
  - NDCG@K  (K = 1, 3, 5)  — position-aware ranking quality
  - Recall@K (K = 1, 3, 5) — binary hit rate
  - MRR      (Mean Reciprocal Rank)

Ground truth: every manifest record maps query_text → page_id.
A retrieval result is considered correct when the returned page_id
matches the ground-truth page_id for that query.

Usage (standalone):
    python scripts/eval_retrieval.py [--top-k 5] [--limit N] [--sample-size N]

Or called programmatically from evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_settings
from src.indexing.vector_store import QdrantVectorStore
from src.models.colpali_encoder import ColPaliEncoder
from src.retrieval.retrieve import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IR Metric helpers (binary relevance — exactly one relevant page per query)
# ---------------------------------------------------------------------------

def ndcg_at_k(relevant_rank: int | None, k: int) -> float:
    """
    NDCG@k when there is exactly 1 relevant document.
    Ideal DCG = 1/log2(1+1) = 1.0  (relevant doc at position 1).
    Actual DCG = 1/log2(rank+1)    (or 0 if not in top-K).
    """
    if relevant_rank is None or relevant_rank > k:
        return 0.0
    import math
    return 1.0 / math.log2(relevant_rank + 1)


def recall_at_k(relevant_rank: int | None, k: int) -> float:
    """Binary recall@k: 1 if relevant doc found in top-k, else 0."""
    if relevant_rank is None or relevant_rank > k:
        return 0.0
    return 1.0


def reciprocal_rank(relevant_rank: int | None) -> float:
    """Reciprocal rank: 1/rank if found, else 0."""
    if relevant_rank is None:
        return 0.0
    return 1.0 / relevant_rank


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def build_retriever(settings) -> Retriever:
    encoder = ColPaliEncoder(model_name=settings.colpali_model)
    store = QdrantVectorStore(
        path=str(settings.qdrant_path),
        collection_name=settings.qdrant_collection,
        vector_size=encoder.vector_dim,
    )
    mv_cache_dir = settings.vidore_manifest_path.parent / "mv_cache"
    return Retriever(encoder=encoder, store=store, mv_cache_dir=mv_cache_dir)


def find_relevant_rank(
    results: list[dict[str, Any]], ground_truth_page_id: str
) -> int | None:
    """
    Return the 1-indexed rank of the ground-truth page_id in the result list,
    or None if it is not present.
    """
    for rank, r in enumerate(results, start=1):
        if r.get("page_id") == ground_truth_page_id:
            return rank
    return None


def run_retrieval_eval(
    settings,
    sample_size: int | None = None,
    top_k: int = 5,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Full retrieval evaluation.

    Args:
        settings:    AppSettings instance.
        sample_size: If set, evaluate on a random stratified sample per dataset.
                     If None, evaluate on all queries.
        top_k:       Maximum rank for retrieval (NDCG/Recall computed at 1, 3, top_k).
        output_dir:  Where to save per-query JSONL and metrics JSON.

    Returns:
        dict with overall and per-dataset metrics.
    """
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    records = load_manifest(settings.vidore_manifest_path)
    logger.info("Loaded %d manifest records.", len(records))

    # Group by dataset (split field) BUT ONLY INCLUDE VALID QUERIES
    by_split: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        query = rec.get("query_text", "").strip()
        gt_page_id = rec.get("page_id", "")
        if query and gt_page_id:
            by_split[rec["split"]].append(rec)

    # Stratified sample if requested
    if sample_size is not None:
        per_split = max(1, sample_size // len(by_split))
        sampled: list[dict] = []
        for split_records in by_split.values():
            pool = split_records[:]
            random.shuffle(pool)
            sampled.extend(pool[:per_split])
        eval_records = sampled
        logger.info(
            "Evaluating on %d sampled queries (%d per dataset).",
            len(eval_records),
            per_split,
        )
    else:
        eval_records = records
        logger.info("Evaluating on all %d queries.", len(eval_records))

    # Load retriever (heavy — do once)
    logger.info("Loading retrieval model …")
    retriever = build_retriever(settings)

    # Output setup
    if output_dir is None:
        output_dir = REPO_ROOT / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "retrieval_detail.jsonl"

    # Accumulate per-split metrics
    split_acc: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    global_acc: dict[str, list[float]] = defaultdict(list)

    ks = [1, 3, top_k]

    with detail_path.open("w", encoding="utf-8") as detail_f:
        for i, rec in enumerate(eval_records):
            query = rec.get("query_text", "").strip()
            gt_page_id = rec.get("page_id", "")
            split = rec.get("split", "unknown")

            if not query or not gt_page_id:
                logger.debug("Skipping record %d — empty query or page_id.", i)
                continue

            t0 = time.perf_counter()
            try:
                results = retriever.retrieve(query, top_k=top_k)
                result_dicts = [r.to_dict() for r in results]
            except Exception as exc:
                logger.warning("Retrieval failed for query %d: %s", i, exc)
                continue
            elapsed = time.perf_counter() - t0

            rank = find_relevant_rank(result_dicts, gt_page_id)

            row: dict[str, Any] = {
                "query_id": rec.get("query_id"),
                "split": split,
                "query": query,
                "gt_page_id": gt_page_id,
                "relevant_rank": rank,
                "retrieval_time_s": round(elapsed, 3),
            }
            for k in ks:
                row[f"ndcg@{k}"] = ndcg_at_k(rank, k)
                row[f"recall@{k}"] = recall_at_k(rank, k)
            row["rr"] = reciprocal_rank(rank)
            detail_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Accumulate
            for k in ks:
                split_acc[split][f"ndcg@{k}"].append(row[f"ndcg@{k}"])
                split_acc[split][f"recall@{k}"].append(row[f"recall@{k}"])
                global_acc[f"ndcg@{k}"].append(row[f"ndcg@{k}"])
                global_acc[f"recall@{k}"].append(row[f"recall@{k}"])
            split_acc[split]["rr"].append(row["rr"])
            global_acc["rr"].append(row["rr"])

            if (i + 1) % 50 == 0:
                logger.info(
                    "Progress: %d/%d | NDCG@%d so far: %.4f",
                    i + 1,
                    len(eval_records),
                    top_k,
                    float(__import__("numpy").mean(global_acc[f"ndcg@{top_k}"])),
                )

    # Build results dict
    import numpy as np

    def _agg(values: list[float]) -> dict[str, float]:
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "n": len(values),
        }

    per_dataset: dict[str, Any] = {}
    for split, metrics in split_acc.items():
        per_dataset[split] = {
            **{f"ndcg@{k}": _agg(metrics[f"ndcg@{k}"]) for k in ks},
            **{f"recall@{k}": _agg(metrics[f"recall@{k}"]) for k in ks},
            "mrr": _agg(metrics["rr"]),
        }

    overall: dict[str, Any] = {
        **{f"ndcg@{k}": _agg(global_acc[f"ndcg@{k}"]) for k in ks},
        **{f"recall@{k}": _agg(global_acc[f"recall@{k}"]) for k in ks},
        "mrr": _agg(global_acc["rr"]),
        "total_queries_evaluated": len(global_acc["rr"]),
    }

    results = {
        "eval_type": "retrieval",
        "top_k": top_k,
        "model": settings.colpali_model,
        "overall": overall,
        "per_dataset": per_dataset,
    }

    metrics_path = output_dir / "retrieval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Retrieval evaluation complete. Results saved to %s", output_dir)
    logger.info(
        "Overall NDCG@%d: %.4f | Recall@%d: %.4f | MRR: %.4f",
        top_k,
        overall[f"ndcg@{top_k}"]["mean"],
        top_k,
        overall[f"recall@{top_k}"]["mean"],
        overall["mrr"]["mean"],
    )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of queries to evaluate (None = all). E.g. 100 for a quick run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write evaluation outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = get_settings()
    run_retrieval_eval(
        settings=settings,
        sample_size=args.sample_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
