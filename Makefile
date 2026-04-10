.PHONY: setup prepare run

setup:
	python -m pip install -r requirements.txt

prepare:
	python scripts/prepare_vidore.py

run:
	flask run
