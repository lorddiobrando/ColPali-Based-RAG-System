.PHONY: setup download prepare build_index run

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
