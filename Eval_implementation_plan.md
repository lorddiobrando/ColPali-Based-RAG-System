# Evaluation Framework for ColPali RAG System

Implement a comprehensive evaluation pipeline that measures retrieval quality, answer generation quality, and factual grounding across the 10 ViDoRe benchmark datasets (8,443 indexed records).

## Data Landscape

The manifest already contains ground truth that we can leverage:

| Dataset | Records | Ground Truth Available |
|---------|---------|----------------------|
| arxivqa_test_subsampled | 500 | ✅ MCQ answer letter + options |
| docvqa_test_subsampled | 500 | ❌ answer is null |
| infovqa_test_subsampled | 500 | ❌ answer is null |
| tabfquad_test_subsampled | 280 | ❌ no answer field |
| tatdqa_test | 1,663 | ⚠️ answer is empty string |
| shiftproject_test | 1,000 | ✅ answer (list of text spans) |
| syntheticDocQA (×4 domains) | 4,000 | ✅ answer (list of text spans) |

**Key insight**: Every record maps a `query_text` → `page_id` (the page the query was created from). This gives us **retrieval ground truth for all 8,443 queries** regardless of whether a text answer exists.

## Proposed Changes

### Evaluation Module

#### [NEW] [evaluate.py](file:///d:/ColPali-Based-RAG-System/scripts/evaluate.py)

Main evaluation orchestrator script. Accepts CLI args (`--eval-mode retrieval|generation|grounding|all`, `--sample-size`, `--top-k`). Writes results to `data/evaluation/` directory.

#### [NEW] [eval_retrieval.py](file:///d:/ColPali-Based-RAG-System/scripts/eval_retrieval.py)

**Retrieval quality evaluation** — runs on the full 8,443-query corpus.

For each query in the manifest:
1. Run the retrieval pipeline (encode query → cosine candidates → MaxSim rerank)
2. Check whether the ground-truth `page_id` appears in the top-K results
3. Compute standard IR metrics:
   - **NDCG@5** — primary ViDoRe metric (position-aware ranking quality)
   - **Recall@1, @3, @5** — does the correct page appear in top K?
   - **MRR** (Mean Reciprocal Rank) — average 1/rank of first correct result

Reports per-dataset breakdown and overall aggregate. Saves detailed per-query results to JSONL for error analysis.

> [!IMPORTANT]
> Since the ViDoRe datasets map each query to exactly one relevant page, we use **binary relevance** (relevant=1 for ground-truth page, 0 for all others). This aligns with the standard ViDoRe V1 evaluation protocol.

#### [NEW] [eval_generation.py](file:///d:/ColPali-Based-RAG-System/scripts/eval_generation.py)

**Answer quality evaluation** — runs on a stratified sample of ~50 queries (5 per dataset) where ground-truth answers exist (arxivqa, shiftproject, syntheticDocQA ×4 = 30 queries with GT answers). For datasets without GT answers (docvqa, infovqa, tabfquad, tatdqa), we evaluate generation but skip the correctness score.

For each sampled query:
1. Run full RAG pipeline (retrieval + generation)
2. Send *(query, generated_answer, ground_truth_answer)* to the LLM judge (OpenRouter)
3. The judge scores on a 1–5 Likert scale across three dimensions:
   - **Correctness** — does the answer match the ground truth?
   - **Completeness** — does it cover all relevant information?
   - **Conciseness** — is it free of unnecessary content?

#### [NEW] [eval_grounding.py](file:///d:/ColPali-Based-RAG-System/scripts/eval_grounding.py)

**Factual grounding evaluation** — runs on the same ~50 query sample.

For each query:
1. Run full RAG pipeline
2. Check **citation accuracy**: does the answer cite a `doc_id`/`page_num` that was actually retrieved?
3. Send *(query, generated_answer, retrieved_page_metadata)* to the LLM judge with a grounding-specific prompt asking:
   - **Faithfulness score (1–5)** — is every claim in the answer supported by the retrieved pages?
   - **Hallucination flag** — does the answer contain information not present in the context?
   - **Citation correctness** — are cited sources actually the ones that contain the relevant info?

---

### Results & Reporting

#### [NEW] [eval_report.py](file:///d:/ColPali-Based-RAG-System/scripts/eval_report.py)

Aggregates all evaluation outputs and produces a final markdown report at `data/evaluation/evaluation_report.md` containing:
- Per-dataset retrieval metrics table
- Aggregate retrieval metrics with confidence intervals
- Generation quality scores (mean/median per dimension)
- Grounding/faithfulness analysis
- Error analysis highlights (worst-performing queries)

---

### Supporting Changes

#### [MODIFY] [requirements.txt](file:///d:/ColPali-Based-RAG-System/requirements.txt)

No new dependencies needed — we only use `numpy` (already present) for metric computation and the existing `requests` library for LLM judge calls. All IR metrics (NDCG, MRR, Recall) are implemented from scratch since they're straightforward with binary relevance.

## Evaluation Query Strategy

Rather than creating a separate query set, we **reuse the ViDoRe benchmark queries** already in the manifest. This is the correct approach because:

1. The ViDoRe benchmark is specifically designed for evaluating document retrieval systems
2. Each query has a known ground-truth page — this is the gold-standard for retrieval eval
3. Several datasets include reference answers — directly usable for generation eval
4. Using the standard benchmark enables comparison with published ColPali results

**Stratified sampling** for generation/grounding eval (5 queries per dataset, total ~50) balances coverage vs. API cost.

## Verification Plan

### Automated Tests
- Run `python scripts/evaluate.py --eval-mode retrieval --top-k 5` and verify NDCG@5 output
- Run `python scripts/evaluate.py --eval-mode all --sample-size 5` for full pipeline test
- Verify that the generated report at `data/evaluation/evaluation_report.md` contains all sections

### Expected Results
- Retrieval NDCG@5 for colSmol-256M should be in the **0.5–0.8** range based on published ViDoRe leaderboard scores for small models
- Generation and grounding scores are qualitative baselines
