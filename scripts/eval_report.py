"""
eval_report.py
--------------
Aggregates all evaluation outputs and generates a polished markdown report
at data/evaluation/evaluation_report.md.

Reads:
  - data/evaluation/retrieval_metrics.json
  - data/evaluation/retrieval_detail.jsonl
  - data/evaluation/generation_metrics.json
  - data/evaluation/generation_detail.jsonl
  - data/evaluation/grounding_metrics.json
  - data/evaluation/grounding_detail.jsonl

Usage (standalone):
    python scripts/eval_report.py [--output-dir path]

Or called programmatically from evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def fmt2(value: float | None) -> str:
    return fmt(value, 2)


def agg_mean(d: dict | None) -> str:
    if d is None:
        return "N/A"
    return fmt(d.get("mean"))


def pct(d: dict | None) -> str:
    """Format a 0–1 mean as percentage."""
    if d is None:
        return "N/A"
    val = d.get("mean")
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def score5(d: dict | None) -> str:
    """Format a 1–5 mean score."""
    if d is None:
        return "N/A"
    val = d.get("mean")
    if val is None:
        return "N/A"
    return f"{val:.2f}/5"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_retrieval_section(metrics: dict | None, detail: list[dict]) -> str:
    if not metrics:
        return "> ⚠️ Retrieval evaluation results not found.\n"

    overall = metrics.get("overall", {})
    per_ds = metrics.get("per_dataset", {})
    top_k = metrics.get("top_k", 5)
    model = metrics.get("model", "unknown")
    n = overall.get("total_queries_evaluated", 0)

    lines = []
    lines.append(f"**Model:** `{model}`  |  **Queries evaluated:** {n}  |  **top_k:** {top_k}\n")

    # Overall summary table
    lines.append("### Overall Metrics\n")
    lines.append(f"| Metric | Score |")
    lines.append(f"|--------|-------|")
    lines.append(f"| NDCG@1 | {agg_mean(overall.get('ndcg@1'))} |")
    lines.append(f"| NDCG@3 | {agg_mean(overall.get('ndcg@3'))} |")
    lines.append(f"| NDCG@{top_k} | {agg_mean(overall.get(f'ndcg@{top_k}'))} |")
    lines.append(f"| Recall@1 | {pct(overall.get('recall@1'))} |")
    lines.append(f"| Recall@3 | {pct(overall.get('recall@3'))} |")
    lines.append(f"| Recall@{top_k} | {pct(overall.get(f'recall@{top_k}'))} |")
    lines.append(f"| MRR | {agg_mean(overall.get('mrr'))} |")
    lines.append("")

    # Per-dataset table
    lines.append("### Per-Dataset Breakdown\n")
    header = f"| Dataset | N | NDCG@{top_k} | Recall@1 | Recall@{top_k} | MRR |"
    sep    = f"|---------|---|-----------|---------|------------|-----|"
    lines.append(header)
    lines.append(sep)

    for ds_key, ds_metrics in sorted(per_ds.items()):
        # Shorten display name: strip "vidore/" prefix and ":test" suffix
        ds_name = ds_key.replace("vidore/", "").replace(":test", "")
        n_ds = (ds_metrics.get(f"ndcg@{top_k}") or {}).get("n", "?")
        ndcg = agg_mean(ds_metrics.get(f"ndcg@{top_k}"))
        r1   = pct(ds_metrics.get("recall@1"))
        rk   = pct(ds_metrics.get(f"recall@{top_k}"))
        mrr  = agg_mean(ds_metrics.get("mrr"))
        lines.append(f"| {ds_name} | {n_ds} | {ndcg} | {r1} | {rk} | {mrr} |")
    lines.append("")

    # Worst-performing queries (bottom 10 by NDCG@top_k)
    if detail:
        ndcg_key = f"ndcg@{top_k}"
        sorted_detail = sorted(detail, key=lambda x: x.get(ndcg_key, 1.0))
        lines.append("### Hardest Queries (bottom 10 by NDCG@5)\n")
        lines.append("| Query (truncated) | Dataset | Rank | NDCG@5 |")
        lines.append("|-------------------|---------|------|--------|")
        for row in sorted_detail[:10]:
            q = row.get("query", "")[:60].replace("|", "/")
            ds = row.get("split", "").replace("vidore/", "").replace(":test", "")
            rank = row.get("relevant_rank", "—")
            score = fmt(row.get(ndcg_key))
            lines.append(f"| {q}… | {ds} | {rank} | {score} |")
        lines.append("")

    return "\n".join(lines)


def build_generation_section(metrics: dict | None, detail: list[dict]) -> str:
    if not metrics:
        return "> ⚠️ Generation evaluation results not found.\n"

    overall = metrics.get("overall", {})
    per_ds = metrics.get("per_dataset", {})
    top_k = metrics.get("top_k", 5)
    sample_size = metrics.get("sample_size", "?")
    judge_model = metrics.get("judge_model", "unknown")
    n = overall.get("total_queries_evaluated", 0)

    lines = []
    lines.append(
        f"**Judge model:** `{judge_model}`  |  **Queries evaluated:** {n}  |  "
        f"**Sample size:** {sample_size}  |  **top_k:** {top_k}\n"
    )

    lines.append("### Overall Quality Scores (LLM-as-Judge, 1–5 scale)\n")
    lines.append("| Dimension | Mean Score | Std |")
    lines.append("|-----------|-----------|-----|")
    for metric, label in [
        ("correctness", "Correctness (vs. ground truth)"),
        ("completeness", "Completeness"),
        ("conciseness", "Conciseness"),
    ]:
        agg = overall.get(metric)
        if agg:
            lines.append(
                f"| {label} | {score5(agg)} | ±{fmt2(agg.get('std'))} |"
            )
        else:
            lines.append(f"| {label} | N/A | — |")
    lines.append("")

    # Per-dataset table
    if per_ds:
        lines.append("### Per-Dataset Generation Quality\n")
        lines.append("| Dataset | Correctness | Completeness | Conciseness |")
        lines.append("|---------|------------|-------------|------------|")
        for ds_key, ds_metrics in sorted(per_ds.items()):
            ds_name = ds_key.replace("vidore/", "").replace(":test", "")
            corr = score5(ds_metrics.get("correctness"))
            comp = score5(ds_metrics.get("completeness"))
            conc = score5(ds_metrics.get("conciseness"))
            lines.append(f"| {ds_name} | {corr} | {comp} | {conc} |")
        lines.append("")

    # Sample of worst answers
    if detail:
        scored = [r for r in detail if r.get("correctness") is not None]
        if scored:
            scored.sort(key=lambda x: x.get("correctness", 5))
            lines.append("### Example Low-Correctness Answers (sampled)\n")
            for row in scored[:3]:
                q = row.get("query", "")[:80]
                gen = row.get("generated_answer", "")[:200].replace("\n", " ")
                gt = (row.get("ground_truth_answer") or "N/A")[:120]
                corr = row.get("correctness", "N/A")
                reason = row.get("reasoning", "")[:150]
                lines.append(f"**Query:** {q}")
                lines.append(f"> **Generated:** {gen}…")
                lines.append(f"> **Ground truth:** {gt}")
                lines.append(f"> **Correctness:** {corr}/5 — {reason}")
                lines.append("")

    return "\n".join(lines)


def build_grounding_section(metrics: dict | None, detail: list[dict]) -> str:
    if not metrics:
        return "> ⚠️ Grounding evaluation results not found.\n"

    overall = metrics.get("overall", {})
    per_ds = metrics.get("per_dataset", {})
    judge_model = metrics.get("judge_model", "unknown")
    n = overall.get("total_queries_evaluated", 0)

    lines = []
    lines.append(f"**Judge model:** `{judge_model}`  |  **Queries evaluated:** {n}\n")

    lines.append("### Overall Grounding Scores\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Faithfulness (1–5) | {score5(overall.get('faithfulness'))} |")
    lines.append(f"| Hallucination Rate | {pct(overall.get('hallucination_rate'))} |")
    lines.append(f"| Citation Accuracy (1–5) | {score5(overall.get('citation_accuracy'))} |")
    lines.append(f"| Answers with Any Citation | {pct(overall.get('citation_rate'))} |")
    lines.append("")

    if per_ds:
        lines.append("### Per-Dataset Grounding\n")
        lines.append("| Dataset | Faithfulness | Hallucination Rate | Citation Accuracy |")
        lines.append("|---------|------------|------------------|-----------------|")
        for ds_key, ds_metrics in sorted(per_ds.items()):
            ds_name = ds_key.replace("vidore/", "").replace(":test", "")
            faith = score5(ds_metrics.get("faithfulness"))
            hall  = pct(ds_metrics.get("hallucination_rate"))
            cit   = score5(ds_metrics.get("citation_accuracy"))
            lines.append(f"| {ds_name} | {faith} | {hall} | {cit} |")
        lines.append("")

    # Hallucination examples
    if detail:
        hallucinated = [
            r for r in detail if r.get("hallucination_detected") is True
        ]
        if hallucinated:
            lines.append(f"### ⚠️ Hallucination Examples ({len(hallucinated)} detected)\n")
            for row in hallucinated[:3]:
                q = row.get("query", "")[:80]
                gen = row.get("generated_answer", "")[:200].replace("\n", " ")
                reason = row.get("reasoning", "")[:150]
                lines.append(f"**Query:** {q}")
                lines.append(f"> **Generated:** {gen}…")
                lines.append(f"> **Judge reasoning:** {reason}")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_report(output_dir: Path) -> Path:
    ret_metrics  = load_json(output_dir / "retrieval_metrics.json")
    gen_metrics  = load_json(output_dir / "generation_metrics.json")
    grd_metrics  = load_json(output_dir / "grounding_metrics.json")

    ret_detail   = load_jsonl(output_dir / "retrieval_detail.jsonl")
    gen_detail   = load_jsonl(output_dir / "generation_detail.jsonl")
    grd_detail   = load_jsonl(output_dir / "grounding_detail.jsonl")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_lines = [
        "# ColPali RAG System — Evaluation Report",
        "",
        f"> Generated: {timestamp}",
        "",
        "This report evaluates the ColPali-based RAG system on three axes:",
        "1. **Retrieval Quality** — Does the system retrieve the correct document page?",
        "2. **Answer Quality** — Is the generated answer correct, complete, and concise?",
        "3. **Factual Grounding** — Is the answer faithful to retrieved context (no hallucinations)?",
        "",
        "**Dataset:** ViDoRe Benchmark (10 datasets, 8,443 indexed records)",
        "**Retriever:** ColPali two-stage pipeline — cosine candidate search + MaxSim reranking",
        "",
        "---",
        "",
        "## 1. Retrieval Quality",
        "",
        build_retrieval_section(ret_metrics, ret_detail),
        "---",
        "",
        "## 2. Answer Generation Quality",
        "",
        build_generation_section(gen_metrics, gen_detail),
        "---",
        "",
        "## 3. Factual Grounding",
        "",
        build_grounding_section(grd_metrics, grd_detail),
        "---",
        "",
        "## Methodology Notes",
        "",
        "- **Retrieval ground truth**: Each ViDoRe query maps to exactly one relevant page (`page_id`).",
        "  NDCG uses binary relevance (1 for the matching page, 0 for all others).",
        "- **Generation & grounding evaluation**: Stratified sample of 50 queries",
        "  (≈5 per dataset). Answers judged by the same OpenRouter LLM used for generation.",
        "- **Hallucination detection**: Both programmatic (regex citation check) and",
        "  LLM-judge based grounding scores are reported.",
        "",
    ]

    report_text = "\n".join(report_lines)
    report_path = output_dir / "evaluation_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report written to: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation report from metric files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "evaluation",
        help="Directory containing metric JSON files and where report will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_report(args.output_dir)
