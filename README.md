# ColPali-Based RAG System

A Flask-based multimodal RAG starter tailored for the ViDoRe benchmark (`vidore/vidore-benchmark`).

## What is implemented now

- Project scaffold for app, data pipeline, and scripts.
- ViDoRe preparation pipeline (`scripts/prepare_vidore.py`) that downloads dataset splits and writes a normalized manifest.
- Config-driven setup via `.env` and `configs/vidore.yaml`.

## Quickstart

1. Create a virtual environment and install dependencies:
   - `python -m venv .venv`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - `pip install -r requirements.txt`
2. Copy environment template:
   - `copy .env.example .env`
3. Prepare ViDoRe artifacts:
   - `python scripts/prepare_vidore.py`
4. Build local vector index:
   - `python scripts/build_index.py --limit 500`
4. Run Flask app:
   - `flask run`

## Project layout

- `app/`: Flask app
- `src/`: core python modules (config, data, retrieval, evaluation)
- `scripts/`: executable entrypoints
- `configs/`: yaml configs
- `data/`: local raw/processed/cache/index artifacts (gitignored in practice)

## Next milestones

- Swap baseline image encoder to true ColPali embeddings.
- Add retrieval endpoint and result citation UI.
- Add benchmark evaluation (Recall@k, MRR, nDCG).
