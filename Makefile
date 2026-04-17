.PHONY: setup download prepare build_index run \
        eval-retrieval eval-generation eval-grounding eval-all eval-quick eval-report

setup:
	python -m pip install -r requirements.txt

download:
	python scripts/download_model.py

prepare:
	python scripts/prepare_vidore.py

build_index:
	python scripts/build_index.py

run:
	flask run

# ── Evaluation targets ──────────────────────────────────────────────────────

# Full retrieval eval on all 8,443 queries (slow — loads model + runs inference)
eval-retrieval:
	python scripts/evaluate.py --eval-mode retrieval --top-k 5 --skip-report

# Generation quality on 50 stratified queries (requires OPENROUTER_API_KEY)
eval-generation:
	python scripts/evaluate.py --eval-mode generation --gen-sample-size 50 --top-k 5 --skip-report

# Factual grounding on 50 stratified queries (requires OPENROUTER_API_KEY)
eval-grounding:
	python scripts/evaluate.py --eval-mode grounding --gen-sample-size 50 --top-k 5 --skip-report

# Full evaluation: retrieval + generation + grounding + report
eval-all:
	python scripts/evaluate.py --eval-mode all --gen-sample-size 50 --top-k 5

# Quick smoke-test: 50 queries retrieval, 10 queries gen/grounding
eval-quick:
	python scripts/evaluate.py --eval-mode all --sample-size 50 --gen-sample-size 10 --top-k 5

# Regenerate the markdown report from existing metric JSON files
eval-report:
	python scripts/eval_report.py
