## Churn Prediction in Telecom Industry

This repository implements a complete machine learning pipeline to predict telecom customer churn using classic supervised learning algorithms. It includes data download from Kaggle, preprocessing, model training, evaluation, and CLI entry points. The project is structured and configured for easy upload to GitHub with CI and minimal tests.

### Dataset
- Source: `https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets`
- Use the provided script to download via the Kaggle API (see below).

### Project Structure
```
.
├── .github/workflows/ci.yml        # CI for lint and tests
├── .gitignore
├── .gitattributes
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml                  # Tooling config (black, isort, flake8, mypy minimal)
├── requirements.txt
├── scripts/
│   └── download_data.py           # Kaggle download helper
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── cli.py                 # CLI entrypoints
│       ├── config.py              # Paths, constants
│       ├── data.py                # Load & preprocess
│       ├── evaluate.py            # Metrics & plots
│       └── models.py              # Model training
├── tests/
│   └── test_preprocess.py
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── models/.gitkeep
└── reports/.gitkeep
```

### Quickstart
1) Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2) Configure Kaggle API (for first-time setup):
- Create an API token from your Kaggle account settings to download datasets.
- Place `kaggle.json` under `~/.kaggle/kaggle.json` and set permissions `chmod 600 ~/.kaggle/kaggle.json`.

3) Download the dataset to `data/raw/`:
```bash
python scripts/download_data.py
```
This will fetch the dataset and place CSV files under `data/raw/`.

4) Run preprocessing to create train/test splits and a preprocessing pipeline:
```bash
python -m churn.cli preprocess \
  --raw_dir data/raw \
  --processed_dir data/processed
```

5) Train a model (choose from: logistic_regression, random_forest, xgboost):
```bash
python -m churn.cli train \
  --processed_dir data/processed \
  --models_dir models \
  --model_name random_forest
```

6) Evaluate the trained model and generate reports/plots:
```bash
python -m churn.cli evaluate \
  --processed_dir data/processed \
  --models_dir models \
  --reports_dir reports \
  --model_name random_forest
```

### Makefile shortcuts
Common tasks are available via Make targets:
```bash
make install
make download
make preprocess
make train MODEL=random_forest
make evaluate MODEL=random_forest
```

### Notes
- The code automatically detects the target label (`Churn`) and converts to binary 0/1.
- It handles common Telco churn dataset quirks (e.g., `TotalCharges` as string). 
- Outputs include metrics (accuracy, precision, recall, f1, roc_auc), confusion matrix, ROC and PR curves.

### License
MIT License. See `LICENSE` for details.


