"""
eval_generation.py
------------------
Answer quality evaluation for the ColPali RAG system.

Uses an LLM-as-judge (via OpenRouter) to score generated answers on:
  - Correctness  (1–5): Does the answer match the ground truth?
  - Completeness (1–5): Does it capture all key information?
  - Conciseness  (1–5): Is it free of hallucinated or irrelevant padding?

Stratified sample: 5 queries per dataset (configurable via --sample-size).
Only queries that have a ground-truth answer in the manifest are scored for
Correctness; the others are scored for Completeness & Conciseness only.

Usage (standalone):
    python scripts/eval_generation.py [--sample-size 50] [--top-k 5]

Or called programmatically from evaluate.py.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
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
# Ground-truth answer extraction
# ---------------------------------------------------------------------------

def extract_gt_answer(rec: dict[str, Any]) -> str | None:
    """
    Extract the ground-truth answer from the raw payload.
    Handles: plain string, list of strings, MCQ letter, None, empty.
    """
    raw = rec.get("raw", {})
    answer = raw.get("answer")

    if answer is None:
        return None

    if isinstance(answer, list):
        # List of text spans → join as bullet points
        return "; ".join(str(a).strip() for a in answer if str(a).strip())

    if isinstance(answer, str):
        stripped = answer.strip()
        if not stripped:
            return None
        # Attempt to parse a Python list literal (shiftproject / syntheticDocQA)
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return "; ".join(str(a).strip() for a in parsed if str(a).strip())
            except (ValueError, SyntaxError):
                pass
        return stripped

    return None


def has_valid_gt(rec: dict[str, Any]) -> bool:
    answer = extract_gt_answer(rec)
    return bool(answer)


# ---------------------------------------------------------------------------
# LLM-as-judge prompt & call
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the quality of answers produced by a "
    "document-based question-answering system. You will receive a question, the system's "
    "generated answer, and optionally a ground-truth reference answer. "
    "Evaluate the generated answer strictly and objectively. "
    "Respond ONLY with a valid JSON object and nothing else."
)


def _build_judge_prompt(
    query: str,
    generated_answer: str,
    ground_truth: str | None,
    has_gt: bool,
) -> str:
    gt_block = (
        f"\nGround-truth reference answer: {ground_truth}"
        if has_gt and ground_truth
        else "\n(No ground-truth reference answer available for this query.)"
    )

    correctness_instruction = (
        '"correctness": <integer 1-5> (1=completely wrong, 5=fully correct; '
        "compare to the ground-truth reference),"
        if has_gt
        else '"correctness": null (no ground-truth available),'
    )

    return (
        f"Question: {query}\n"
        f"Generated answer: {generated_answer}"
        f"{gt_block}\n\n"
        "Evaluate the generated answer on the following dimensions. "
        "Return a JSON object with these exact keys:\n"
        "{\n"
        f"  {correctness_instruction}\n"
        '  "completeness": <integer 1-5> (1=misses almost everything, 5=fully complete),\n'
        '  "conciseness": <integer 1-5> (1=extremely verbose/padded, 5=perfectly concise),\n'
        '  "reasoning": "<one sentence explanation of your scores>"\n'
        "}"
    )


def call_judge(
    settings,
    query: str,
    generated_answer: str,
    ground_truth: str | None,
    has_gt: bool,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Call the OpenRouter LLM judge and return parsed scores.
    Returns a dict with correctness, completeness, conciseness, reasoning.
    """
    import requests

    prompt = _build_judge_prompt(query, generated_answer, ground_truth, has_gt)

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
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
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
                "correctness": parsed.get("correctness"),
                "completeness": parsed.get("completeness"),
                "conciseness": parsed.get("conciseness"),
                "reasoning": parsed.get("reasoning", ""),
            }
        except Exception as exc:
            logger.warning("Judge call attempt %d failed: %s", attempt, exc)
            time.sleep(2 * attempt)

    return {
        "correctness": None,
        "completeness": None,
        "conciseness": None,
        "reasoning": "Judge call failed.",
    }


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def run_generation_eval(
    settings,
    sample_size: int = 50,
    top_k: int = 5,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generation quality evaluation via LLM-as-judge.

    Args:
        settings:    AppSettings instance.
        sample_size: Total queries to evaluate (stratified across datasets).
        top_k:       Top-k pages retrieved before generation.
        output_dir:  Directory to write output files.
        seed:        Random seed for reproducibility.

    Returns:
        dict with overall and per-dataset generation quality metrics.
    """
    from app.services.generation_service import generate_answer
    from app.services.retrieval_service import run_retrieval

    if not settings.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not set — cannot run generation evaluation.")
        return {"eval_type": "generation", "error": "No API key"}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Load manifest
    records: list[dict[str, Any]] = []
    with settings.vidore_manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Group by split
    by_split: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_split[rec["split"]].append(rec)

    # Stratified sample — prefer records with GT answers to maximise scoring coverage
    rng = random.Random(seed)
    per_split = max(1, sample_size // len(by_split))
    eval_records: list[dict] = []
    for split, split_recs in by_split.items():
        gt_recs = [r for r in split_recs if has_valid_gt(r)]
        no_gt_recs = [r for r in split_recs if not has_valid_gt(r)]
        # Prefer records with GT, top up with no-GT if needed
        selected = rng.sample(gt_recs, min(per_split, len(gt_recs)))
        remaining = per_split - len(selected)
        if remaining > 0 and no_gt_recs:
            selected += rng.sample(no_gt_recs, min(remaining, len(no_gt_recs)))
        eval_records.extend(selected)

    logger.info("Generation eval: %d queries sampled (%d per dataset).", len(eval_records), per_split)

    # Output setup
    if output_dir is None:
        output_dir = REPO_ROOT / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "generation_detail.jsonl"

    split_acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    global_acc: dict[str, list] = defaultdict(list)

    with detail_path.open("w", encoding="utf-8") as detail_f:
        for i, rec in enumerate(eval_records):
            query = rec.get("query_text", "").strip()
            split = rec.get("split", "unknown")
            gt_answer = extract_gt_answer(rec)
            has_gt = bool(gt_answer)

            if not query:
                continue

            logger.info("[%d/%d] %s — %s", i + 1, len(eval_records), split, query[:80])

            # 1. Retrieve
            try:
                retrieval = run_retrieval(settings, query, top_k=top_k)
                retrieval_results = retrieval["results"]
            except Exception as exc:
                logger.warning("Retrieval failed: %s", exc)
                continue

            # 2. Generate
            try:
                generation = generate_answer(settings, query, retrieval_results)
                generated_answer = generation.get("answer") or ""
                gen_time = generation.get("timing", {}).get("generation_seconds", 0)
            except Exception as exc:
                logger.warning("Generation failed: %s", exc)
                continue

            if not generated_answer:
                logger.warning("Empty answer for query %d — skipping.", i)
                continue

            # 3. Judge
            scores = call_judge(
                settings, query, generated_answer, gt_answer, has_gt
            )

            row = {
                "query_id": rec.get("query_id"),
                "split": split,
                "query": query,
                "generated_answer": generated_answer,
                "ground_truth_answer": gt_answer,
                "has_ground_truth": has_gt,
                "generation_time_s": gen_time,
                **scores,
            }
            detail_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Accumulate numeric scores
            for metric in ("correctness", "completeness", "conciseness"):
                val = scores[metric]
                if val is not None:
                    split_acc[split][metric].append(float(val))
                    global_acc[metric].append(float(val))

    # Aggregate
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
        per_dataset[split] = {
            m: _agg(v) for m, v in metrics.items()
        }

    overall: dict[str, Any] = {
        m: _agg(v) for m, v in global_acc.items()
    }
    overall["total_queries_evaluated"] = len(eval_records)

    results = {
        "eval_type": "generation",
        "top_k": top_k,
        "sample_size": sample_size,
        "model": settings.colpali_model,
        "judge_model": settings.openrouter_model,
        "overall": overall,
        "per_dataset": per_dataset,
    }

    metrics_path = output_dir / "generation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Generation evaluation complete → %s", metrics_path)
    for metric, agg in overall.items():
        if isinstance(agg, dict):
            logger.info("  %s: mean=%.2f ± %.2f (n=%d)", metric, agg["mean"], agg["std"], agg["n"])

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generation quality via LLM judge.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = get_settings()
    run_generation_eval(
        settings=settings,
        sample_size=args.sample_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
        seed=args.seed,
    )
