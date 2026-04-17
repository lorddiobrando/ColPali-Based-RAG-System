# ColPali-Based RAG System

A multimodal Retrieval-Augmented Generation system that uses **ColSmol** (or alternatively ColQwen2) with **late interaction scoring** to search through document pages as images — no OCR needed.

Built on the [ViDoRe benchmark](https://huggingface.co/collections/vidore/vidore-benchmark-667173f98e70a1c0fa4db00d) dataset.

## Features

- **ColSmol Encoder** — Lightweight Vision-language model that "sees" PDF pages and produces multi-vector embeddings
- **Late Interaction (MaxSim)** — ColBERT-style matching for precise retrieval: each query token finds its best visual patch match
- **Two-Stage Retrieval** — Mean-pooled cosine search for candidates → MaxSim reranking for precision
- **RAG Generation** — Answer questions with cited page references via OpenRouter API (Base64 Native Visual Context)
- **Retrieval-Only Mode** — Works without an LLM API key for pure document search
- **Polished Flask UI** — Sleek Natural/Wood-themed interface with page previews, full-screen lightbox, debug panel, and responsive design

## Architecture

```
Query → ColSmol encode → Cosine search (Qdrant) → Top-50 candidates
        → MaxSim rerank (cached multi-vectors) → Top-K results
        → OpenRouter LLM → Answer + Citations
```

## Quickstart

### 1. Environment Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env
# Edit .env: set OPENROUTER_API_KEY for RAG mode (optional)
```

### 3. Setup Models & Dataset

```bash
# Optional: Pre-download model explicitly (bypasses HF xet bugs)
python scripts/download_model.py

# Download and parse ViDoRe images
python scripts/prepare_vidore.py
```

### 4. Build Vector Index

```bash
# Full index (GPU recommended — RTX 3060 or better):
python scripts/build_index.py

# Quick demo with limited pages:
python scripts/build_index.py --limit 100

# Reset and re-index:
python scripts/build_index.py --reset --limit 100
```

### 5. Run the App

```bash
flask run
```

Open [http://localhost:5000](http://localhost:5000) and start querying.

### 6. Evaluation

The system includes a comprehensive LLM-as-a-judge evaluation framework running on the standard ViDoRe benchmark. Evaluation tests Retrieval Quality, Answer Generation Quality, and Factual Grounding (Hallucination detection).

```bash
# Run quick sample evaluation (50 queries total)
make eval-quick

# Full retrieval evaluation on all 8,443 queries
make eval-retrieval

# Generation & Grounding evaluation (Requires OPENROUTER_API_KEY)
make eval-generation
make eval-grounding

# Run all axes and generate a Markdown report
make eval-all
```

## Project Layout

```
├── app/                    # Flask application
│   ├── routes.py           # API endpoints
│   ├── services/           # Retrieval + generation services
│   ├── templates/          # Jinja2 HTML templates
│   └── static/             # CSS + JS assets
├── src/                    # Core modules
│   ├── config/             # Settings and env config
│   ├── data/               # Dataset loaders and schemas
│   ├── models/             # ColSmol/ColQwen2 encoder + baseline
│   ├── indexing/           # Qdrant vector store + index builder
│   └── retrieval/          # Two-stage retrieval pipeline
├── scripts/                # CLI entrypoints
│   ├── download_model.py   # Resilient model downloader
│   ├── prepare_vidore.py   # Download and normalize ViDoRe
│   └── build_index.py      # Build vector index
├── configs/                # YAML configs
│   └── vidore.yaml         # Dataset split + processing config
└── data/                   # Local artifacts (gitignored)
    ├── processed/          # Manifest + exported images
    └── indexes/            # Qdrant database files
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `COLPALI_MODEL` | `vidore/colSmol-256M` | HuggingFace model ID |
| `ENCODER_TYPE` | `colpali` | `colpali` or `baseline` |
| `QDRANT_PATH` | `./data/indexes/qdrant` | Local Qdrant storage |
| `OPENROUTER_API_KEY` | (empty) | Set to enable RAG generation |
| `OPENROUTER_MODEL` | `google/gemma-4-26b-a4b-it` | Multi-modal LLM model for generation |

## Next Milestones

- Text-only OCR baseline comparison
- Async ingestion for large corpora
- Integration with external knowledge graphs
