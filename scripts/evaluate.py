"""
evaluate.py
-----------
Top-level evaluation orchestrator for the ColPali RAG system.

Runs any combination of:
  - retrieval  : NDCG@K, Recall@K, MRR on ViDoRe ground-truth query→page mappings
  - generation : LLM-as-judge answer quality (correctness, completeness, conciseness)
  - grounding  : Factual faithfulness, hallucination detection, citation accuracy
  - all        : All three, then generate the final markdown report

Usage examples:
    # Full evaluation (all axes) on a 50-query stratified sample for gen/grounding
    python scripts/evaluate.py --eval-mode all

    # Quick retrieval-only run on 200 queries
    python scripts/evaluate.py --eval-mode retrieval --sample-size 200 --top-k 5

    # Full retrieval eval on all 8443 queries
    python scripts/evaluate.py --eval-mode retrieval

    # Only run the report (if metric files already exist)
    python scripts/evaluate.py --eval-mode report
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ColPali RAG System — Evaluation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--eval-mode",
        choices=["retrieval", "generation", "grounding", "all", "report"],
        default="all",
        help="Which evaluation(s) to run (default: all).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of pages to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "For retrieval eval: number of queries to evaluate (default: all 8443). "
            "For generation/grounding: number of queries to sample (default: 50)."
        ),
    )
    parser.add_argument(
        "--gen-sample-size",
        type=int,
        default=50,
        help="Number of queries for generation & grounding eval (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write all evaluation outputs (default: data/evaluation/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip automatic report generation after evaluation.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    from src.config.settings import get_settings

    settings = get_settings()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.eval_mode
    top_k = args.top_k
    seed = args.seed

    results: dict = {}

    # ------------------------------------------------------------------
    # Retrieval evaluation
    # ------------------------------------------------------------------
    if mode in ("retrieval", "all"):
        logger.info("=" * 60)
        logger.info("STEP 1/3 — Retrieval Evaluation")
        logger.info("=" * 60)
        t0 = time.perf_counter()

        from scripts.eval_retrieval import run_retrieval_eval

        # For retrieval: use --sample-size directly (None = all queries)
        ret_sample = args.sample_size  # None → evaluate all 8443 queries
        ret_results = run_retrieval_eval(
            settings=settings,
            sample_size=ret_sample,
            top_k=top_k,
            output_dir=output_dir,
        )
        results["retrieval"] = ret_results
        elapsed = time.perf_counter() - t0
        logger.info("Retrieval evaluation finished in %.1f s.", elapsed)

    # ------------------------------------------------------------------
    # Generation quality evaluation
    # ------------------------------------------------------------------
    if mode in ("generation", "all"):
        logger.info("=" * 60)
        logger.info("STEP 2/3 — Generation Quality Evaluation")
        logger.info("=" * 60)
        t0 = time.perf_counter()

        from scripts.eval_generation import run_generation_eval

        gen_sample = args.sample_size if args.sample_size else args.gen_sample_size
        gen_results = run_generation_eval(
            settings=settings,
            sample_size=gen_sample,
            top_k=top_k,
            output_dir=output_dir,
            seed=seed,
        )
        results["generation"] = gen_results
        elapsed = time.perf_counter() - t0
        logger.info("Generation evaluation finished in %.1f s.", elapsed)

    # ------------------------------------------------------------------
    # Grounding evaluation
    # ------------------------------------------------------------------
    if mode in ("grounding", "all"):
        logger.info("=" * 60)
        logger.info("STEP 3/3 — Factual Grounding Evaluation")
        logger.info("=" * 60)
        t0 = time.perf_counter()

        from scripts.eval_grounding import run_grounding_eval

        grd_sample = args.sample_size if args.sample_size else args.gen_sample_size
        grd_results = run_grounding_eval(
            settings=settings,
            sample_size=grd_sample,
            top_k=top_k,
            output_dir=output_dir,
            seed=seed,
        )
        results["grounding"] = grd_results
        elapsed = time.perf_counter() - t0
        logger.info("Grounding evaluation finished in %.1f s.", elapsed)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    if (mode in ("report", "all") or not args.skip_report) and mode != "retrieval":
        logger.info("=" * 60)
        logger.info("Generating evaluation report …")
        logger.info("=" * 60)
        from scripts.eval_report import generate_report
        report_path = generate_report(output_dir)
        logger.info("Report ready: %s", report_path)
    elif mode == "retrieval" and not args.skip_report:
        # Still generate/update report even for retrieval-only run
        from scripts.eval_report import generate_report
        generate_report(output_dir)

    # ------------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    ret = results.get("retrieval", {})
    if ret and "overall" in ret:
        ov = ret["overall"]
        k  = ret.get("top_k", 5)
        logger.info(
            "Retrieval  | NDCG@%d=%.4f | Recall@%d=%.4f | MRR=%.4f | N=%s",
            k, ov.get(f"ndcg@{k}", {}).get("mean", 0),
            k, ov.get(f"recall@{k}", {}).get("mean", 0),
            ov.get("mrr", {}).get("mean", 0),
            ov.get("total_queries_evaluated", "?"),
        )

    gen = results.get("generation", {})
    if gen and "overall" in gen:
        ov = gen["overall"]
        def _m(key):
            d = ov.get(key)
            return f"{d['mean']:.2f}" if d else "N/A"
        logger.info(
            "Generation | Correctness=%s | Completeness=%s | Conciseness=%s | N=%s",
            _m("correctness"), _m("completeness"), _m("conciseness"),
            ov.get("total_queries_evaluated", "?"),
        )

    grd = results.get("grounding", {})
    if grd and "overall" in grd:
        ov = grd["overall"]
        def _m(key):
            d = ov.get(key)
            return f"{d['mean']:.3f}" if d else "N/A"
        logger.info(
            "Grounding  | Faithfulness=%s | HallucinationRate=%s | CitationAcc=%s | N=%s",
            _m("faithfulness"), _m("hallucination_rate"), _m("citation_accuracy"),
            ov.get("total_queries_evaluated", "?"),
        )

    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    args = parse_args()
    run(args)
