# Project Architecture

## Directory Structure

```
Telecom-Churn-Detection/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── Makefile                       # Build automation
├── pyproject.toml                 # Tool configuration
├── requirements.txt               # Dependencies
├── scripts/
│   └── download_data.py           # Kaggle dataset downloader
├── src/churn/
│   ├── __init__.py
│   ├── cli.py                     # CLI entry point (Click)
│   ├── config.py                  # Constants & paths
│   ├── data.py                    # Data loading & preprocessing
│   ├── evaluate.py                # Metrics & visualizations
│   └── models.py                  # Model training & tuning
├── tests/
│   └── test_preprocess.py         # Unit tests
├── data/
│   ├── raw/                       # Raw CSVs from Kaggle
│   └── processed/                 # Parquet files + preprocessor
├── models/                        # Saved .joblib models
└── reports/                       # Metrics JSON + plots
```

## Data Flow Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│  Kaggle API  │───>│  Raw CSVs    │───>│  Preprocess  │───>│  Parquet   │
│  download    │    │  data/raw/   │    │  clean+split │    │  data/proc │
└─────────────┘    └──────────────┘    └──────────────┘    └────────────┘
                                                                  │
                                           ┌──────────────────────┤
                                           ▼                      ▼
                                    ┌──────────────┐    ┌──────────────┐
                                    │    Train      │    │   Evaluate   │
                                    │  GridSearchCV │    │  Metrics +   │
                                    │  models/      │    │  Plots       │
                                    └──────────────┘    └──────────────┘
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config.py` | Central configuration: paths, constants (`RANDOM_STATE=42`, `TEST_SIZE=0.2`, `TARGET_COLUMN="Churn"`) |
| `data.py` | Load CSVs, clean data, build [[02-Data-Processing/Scikit-Learn Pipelines\|sklearn Pipeline]], split into train/test |
| `models.py` | Define model + param grids, run [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning\|GridSearchCV]], save best model |
| `evaluate.py` | Load model, predict, compute metrics, generate plots |
| `cli.py` | [[01-Python-Fundamentals/Click CLI Framework\|Click]]-based CLI wrapping the above modules |

## Key Design Decisions

1. **Modular separation** - Each concern (data, training, evaluation) in its own module
2. **Parquet for intermediate data** - Faster I/O than CSV, preserves types. See [[01-Python-Fundamentals/PyArrow and Parquet]]
3. **Preprocessor saved as artifact** - Same transformations applied at train and test time (no data leakage)
4. **GridSearchCV with ROC-AUC scoring** - Optimizes for the right metric given [[04-Model-Evaluation/Class Imbalance]]
5. **CLI entry point** - Reproducible runs via `make` commands

## Reproducibility Guarantees

- `RANDOM_STATE = 42` used everywhere (splits, models)
- Stratified train-test split preserves class distribution
- Pinned dependency versions in `requirements.txt`
- CI pipeline ensures linting and tests pass

---

**Related:** [[00-Project-Overview/Dataset and Business Context]] | [[06-DevOps-and-Tooling/Makefile Automation]]
