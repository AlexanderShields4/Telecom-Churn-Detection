PYTHON?=python
PIP?=pip

.PHONY: install download preprocess train evaluate lint test

install:
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) scripts/download_data.py

preprocess:
	$(PYTHON) -m churn.cli preprocess --raw_dir data/raw --processed_dir data/processed

train:
	$(PYTHON) -m churn.cli train --processed_dir data/processed --models_dir models --model_name $(MODEL)

evaluate:
	$(PYTHON) -m churn.cli evaluate --processed_dir data/processed --models_dir models --reports_dir reports --model_name $(MODEL)

lint:
	flake8 src tests

test:
	pytest -q


