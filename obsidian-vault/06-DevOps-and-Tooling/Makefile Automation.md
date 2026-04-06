# Makefile Automation

## What Is It?

A `Makefile` defines a set of tasks (targets) that can be run with `make <target>`. Originally for compiling C/C++, it's widely used for task automation in any project.

## This Project's Makefile

```makefile
install:
    pip install -r requirements.txt

download:
    python scripts/download_data.py

preprocess:
    python -m churn.cli preprocess

train:
    python -m churn.cli train --model_name $(MODEL)

evaluate:
    python -m churn.cli evaluate --model_name $(MODEL)

lint:
    flake8 src tests

test:
    pytest
```

## Usage

```bash
make install                        # Install dependencies
make download                       # Download dataset from Kaggle
make preprocess                     # Clean and split data
make train MODEL=random_forest      # Train a model
make evaluate MODEL=random_forest   # Evaluate the model
make lint                           # Run linter
make test                           # Run tests
```

## Full Workflow

```bash
make install && make download && make preprocess && \
make train MODEL=random_forest && make evaluate MODEL=random_forest
```

## Key Makefile Concepts

| Concept | Example | Meaning |
|---------|---------|---------|
| Target | `install:` | Name of the task |
| Recipe | `pip install ...` | Commands to run (must use TAB indent) |
| Variable | `$(MODEL)` | Parameterized from CLI: `make train MODEL=xgboost` |
| `.PHONY` | `.PHONY: install test` | Declare targets that aren't files |
| Dependencies | `train: preprocess` | Run `preprocess` before `train` |

## Why Use Make for ML Projects?

1. **Reproducibility**: Anyone can run the exact same pipeline
2. **Documentation**: Makefile serves as a runbook
3. **Simplicity**: No need to remember long CLI commands
4. **Composability**: Chain targets with dependencies

## Alternatives

| Tool | When To Use |
|------|------------|
| **Makefile** | Simple task running (this project) |
| **DVC** | ML-specific pipeline with data versioning |
| **Airflow/Prefect** | Complex DAG workflows, scheduling |
| **Luigi** | Data pipeline dependencies |
| **Just** | Modern Make alternative with better UX |

---

**Related:** [[01-Python-Fundamentals/Click CLI Framework]] | [[00-Project-Overview/Project Architecture]]
