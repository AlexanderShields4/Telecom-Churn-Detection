# Click CLI Framework

## What Is It?

Click is a Python package for creating command-line interfaces. It uses decorators to define commands, options, and arguments, making CLI creation clean and composable.

## How It's Used in This Project

```python
import click

@click.group()
def cli():
    """Telecom churn prediction pipeline."""
    pass

@cli.command()
@click.option("--raw_dir", default="data/raw", help="Path to raw data")
@click.option("--processed_dir", default="data/processed", help="Output path")
def preprocess(raw_dir, processed_dir):
    """Load raw CSVs, clean, preprocess, and split."""
    # ... calls data.py functions

@cli.command()
@click.option("--model_name", type=click.Choice(["logistic_regression", "random_forest", "xgboost"]))
@click.option("--cv_folds", default=5, type=int)
def train(processed_dir, models_dir, model_name, cv_folds):
    """Train a model with hyperparameter tuning."""
    # ... calls models.py functions

@cli.command()
def evaluate(processed_dir, models_dir, reports_dir, model_name):
    """Evaluate a trained model and generate reports."""
    # ... calls evaluate.py functions
```

## Key Concepts

### `@click.group()` - Command Groups
Groups multiple commands under one CLI. Run individual commands as subcommands:
```bash
python -m churn.cli preprocess
python -m churn.cli train --model_name xgboost
python -m churn.cli evaluate --model_name xgboost
```

### `@click.option()` - Named Parameters
- `default` - Default value if not provided
- `type` - Type validation (`int`, `float`, `click.Choice([...])`)
- `help` - Help text shown in `--help`

### `@click.argument()` vs `@click.option()`
- **Arguments**: Positional, required, no `--` prefix
- **Options**: Named with `--`, can have defaults, more flexible

### Why Click over argparse?
| Feature | argparse (stdlib) | Click |
|---------|-------------------|-------|
| Syntax | Imperative (add_argument) | Declarative (decorators) |
| Composability | Manual subparser setup | Simple `@group` + `@command` |
| Type validation | Basic | Rich (Choice, Path, IntRange) |
| Testing | Harder | Built-in `CliRunner` |

## Interview Relevance

Click demonstrates understanding of:
- **Decorator pattern** in Python
- **Separation of concerns** - CLI layer is thin, delegates to domain modules
- **User-facing interface design** - Defaults, help text, validation

---

**Related:** [[00-Project-Overview/Project Architecture]] | [[06-DevOps-and-Tooling/Makefile Automation]]
