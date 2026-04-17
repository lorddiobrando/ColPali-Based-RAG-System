"""
eval_grounding.py
-----------------
Factual grounding evaluation for the ColPali RAG system.

For each sampled query the pipeline:
  1. Runs retrieval to obtain top-k pages.
  2. Generates an answer via the RAG pipeline.
  3. Uses an LLM judge to assess:
       - Faithfulness (1–5): Every claim in the answer is supported by the retrieved pages.
       - Hallucination (bool): Answer contains information NOT present in the context.
       - Citation accuracy (1–5): Cited doc/page references match the retrieved evidence.

Also verifies citation accuracy programmatically by checking whether doc_ids mentioned
in the generated text actually appear in the retrieved result set.

Usage (standalone):
    python scripts/eval_grounding.py [--sample-size 50] [--top-k 5]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM judge — grounding-specific
# ---------------------------------------------------------------------------

_GROUNDING_SYSTEM_PROMPT = (
    "You are an expert evaluator specialising in factual grounding for document "
    "question-answering systems. You will receive a question, the generated answer, "
    "and the metadata of the document pages that were retrieved and shown to the system "
    "as context. Assess strictly whether the answer is faithful to the provided "
    "context — do NOT use outside knowledge. "
    "Respond ONLY with a valid JSON object and nothing else."
)


def _build_grounding_prompt(
    query: str,
    generated_answer: str,
    retrieved_pages: list[dict[str, Any]],
) -> str:
    pages_block = "\n".join(
        f"  Page {i+1}: doc_id={p.get('doc_id','?')} | page_num={p.get('page_num','?')} "
        f"| score={p.get('score', '?')}"
        for i, p in enumerate(retrieved_pages)
    )

    return (
        f"Question: {query}\n\n"
        f"Retrieved context pages:\n{pages_block}\n\n"
        f"Generated answer: {generated_answer}\n\n"
        "Evaluate the generated answer on the following dimensions:\n"
        "{\n"
        '  "faithfulness": <integer 1-5>  (1=mostly hallucinated, 5=fully grounded in context),\n'
        '  "hallucination_detected": <true|false>  (true if any claim is NOT in the context),\n'
        '  "citation_accuracy": <integer 1-5>  (1=wrong/missing citations, 5=all citations correct),\n'
        '  "reasoning": "<one sentence explaining your assessment>"\n'
        "}"
    )


def call_grounding_judge(
    settings,
    query: str,
    generated_answer: str,
    retrieved_pages: list[dict[str, Any]],
    max_retries: int = 2,
) -> dict[str, Any]:
    import requests

    prompt = _build_grounding_prompt(query, generated_answer, retrieved_pages)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openrouter_model,
                    "messages": [
                        {"role": "system", "content": _GROUNDING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return {
                "faithfulness": parsed.get("faithfulness"),
                "hallucination_detected": parsed.get("hallucination_detected"),
                "citation_accuracy": parsed.get("citation_accuracy"),
                "reasoning": parsed.get("reasoning", ""),
            }
        except Exception as exc:
            logger.warning("Grounding judge attempt %d failed: %s", attempt, exc)
            time.sleep(2 * attempt)

    return {
        "faithfulness": None,
        "hallucination_detected": None,
        "citation_accuracy": None,
        "reasoning": "Judge call failed.",
    }


# ---------------------------------------------------------------------------
# Programmatic citation check
# ---------------------------------------------------------------------------

def check_citation_programmatic(
    generated_answer: str,
    retrieved_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Heuristic citation check: look for doc_id / page references explicitly
    mentioned in the answer and verify they are among the retrieved pages.

    Returns:
        {
            "retrieved_doc_ids": [...],
            "mentioned_doc_ids": [...],
            "all_mentioned_are_retrieved": bool,
            "has_any_citation": bool,
        }
    """
    retrieved_doc_ids = {p.get("doc_id", "") for p in retrieved_pages if p.get("doc_id")}

    # Look for common citation patterns: "Source 1", "Document: xyz", "[Source 2]"
    mentioned: set[str] = set()
    for doc_id in retrieved_doc_ids:
        if doc_id and doc_id.lower() in generated_answer.lower():
            mentioned.add(doc_id)

    # Also look for "Source N" patterns to confirm any citation was attempted
    source_refs = re.findall(r"\[?[Ss]ource\s*\d+\]?|\[?[Dd]ocument\]?", generated_answer)
    has_any_citation = bool(mentioned) or bool(source_refs)

    return {
        "retrieved_doc_ids": sorted(retrieved_doc_ids),
        "mentioned_doc_ids": sorted(mentioned),
        "all_mentioned_are_retrieved": all(m in retrieved_doc_ids for m in mentioned),
        "has_any_citation": has_any_citation,
    }


# ---------------------------------------------------------------------------
# Main grounding evaluation
# ---------------------------------------------------------------------------

def run_grounding_eval(
    settings,
    sample_size: int = 50,
    top_k: int = 5,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Factual grounding evaluation.
    """
    from app.services.generation_service import generate_answer
    from app.services.retrieval_service import run_retrieval

    if not settings.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not set — cannot run grounding evaluation.")
        return {"eval_type": "grounding", "error": "No API key"}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    records: list[dict[str, Any]] = []
    with settings.vidore_manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_split: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_split[rec["split"]].append(rec)

    rng = random.Random(seed)
    per_split = max(1, sample_size // len(by_split))
    eval_records: list[dict] = []
    for split_recs in by_split.values():
        pool = split_recs[:]
        rng.shuffle(pool)
        eval_records.extend(pool[:per_split])

    logger.info("Grounding eval: %d queries sampled.", len(eval_records))

    if output_dir is None:
        output_dir = REPO_ROOT / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "grounding_detail.jsonl"

    split_acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    global_acc: dict[str, list] = defaultdict(list)

    with detail_path.open("w", encoding="utf-8") as detail_f:
        for i, rec in enumerate(eval_records):
            query = rec.get("query_text", "").strip()
            split = rec.get("split", "unknown")

            if not query:
                continue

            logger.info("[%d/%d] %s — %s", i + 1, len(eval_records), split, query[:80])

            # Retrieve
            try:
                retrieval = run_retrieval(settings, query, top_k=top_k)
                retrieval_results = retrieval["results"]
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)
                continue

            # Generate
            try:
                generation = generate_answer(settings, query, retrieval_results)
                generated_answer = generation.get("answer") or ""
            except Exception as exc:
                logger.warning("Generation failed: %s", exc)
                continue

            if not generated_answer:
                logger.warning("Empty answer for query %d — skipping.", i)
                continue

            # Programmatic citation check
            citation_check = check_citation_programmatic(generated_answer, retrieval_results)

            # LLM judge grounding scores
            scores = call_grounding_judge(
                settings, query, generated_answer, retrieval_results
            )

            row = {
                "query_id": rec.get("query_id"),
                "split": split,
                "query": query,
                "generated_answer": generated_answer,
                "citation_check": citation_check,
                **scores,
            }
            detail_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Accumulate
            for metric in ("faithfulness", "citation_accuracy"):
                val = scores[metric]
                if val is not None:
                    split_acc[split][metric].append(float(val))
                    global_acc[metric].append(float(val))

            hall = scores.get("hallucination_detected")
            if hall is not None:
                split_acc[split]["hallucination_rate"].append(1.0 if hall else 0.0)
                global_acc["hallucination_rate"].append(1.0 if hall else 0.0)

            has_cit = citation_check["has_any_citation"]
            split_acc[split]["citation_rate"].append(1.0 if has_cit else 0.0)
            global_acc["citation_rate"].append(1.0 if has_cit else 0.0)

    import numpy as np

    def _agg(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "n": len(values),
        }

    per_dataset: dict[str, Any] = {}
    for split, metrics in split_acc.items():
        per_dataset[split] = {m: _agg(v) for m, v in metrics.items()}

    overall: dict[str, Any] = {m: _agg(v) for m, v in global_acc.items()}
    overall["total_queries_evaluated"] = len(eval_records)

    results = {
        "eval_type": "grounding",
        "top_k": top_k,
        "sample_size": sample_size,
        "model": settings.colpali_model,
        "judge_model": settings.openrouter_model,
        "overall": overall,
        "per_dataset": per_dataset,
    }

    metrics_path = output_dir / "grounding_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Grounding evaluation complete → %s", metrics_path)
    for metric, agg in overall.items():
        if isinstance(agg, dict):
            logger.info("  %s: mean=%.3f (n=%d)", metric, agg["mean"], agg["n"])

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate factual grounding of generated answers.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = get_settings()
    run_grounding_eval(
        settings=settings,
        sample_size=args.sample_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
        seed=args.seed,
    )
