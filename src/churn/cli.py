from pathlib import Path

import click

from .config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, MODELS_DIR
from .data import preprocess_and_split
from .models import train_model
from .evaluate import evaluate_model


@click.group()
def cli() -> None:
    """Churn prediction CLI."""


@cli.command()
@click.option("--raw_dir", type=click.Path(path_type=Path), default=RAW_DIR, show_default=True)
@click.option(
    "--processed_dir", type=click.Path(path_type=Path), default=PROCESSED_DIR, show_default=True
)
def preprocess(raw_dir: Path, processed_dir: Path) -> None:
    """Load raw CSVs, clean, split and save processed arrays and preprocessor."""
    paths, shapes = preprocess_and_split(raw_dir=raw_dir, processed_dir=processed_dir)
    click.echo("Preprocessing complete. Saved to:")
    click.echo(f" - {paths.X_train_path}")
    click.echo(f" - {paths.X_test_path}")
    click.echo(f" - {paths.y_train_path}")
    click.echo(f" - {paths.y_test_path}")
    click.echo(f" - {paths.preprocessor_path}")


@cli.command()
@click.option(
    "--processed_dir", type=click.Path(path_type=Path), default=PROCESSED_DIR, show_default=True
)
@click.option("--models_dir", type=click.Path(path_type=Path), default=MODELS_DIR, show_default=True)
@click.option(
    "--model_name",
    type=click.Choice(["logistic_regression", "random_forest", "xgboost"], case_sensitive=False),
    default="random_forest",
    show_default=True,
)
@click.option("--cv_folds", type=int, default=5, show_default=True)
def train(processed_dir: Path, models_dir: Path, model_name: str, cv_folds: int) -> None:
    """Train a model with cross-validated hyperparameter search and save the best estimator."""
    result = train_model(processed_dir=processed_dir, models_dir=models_dir, model_name=model_name, cv_folds=cv_folds)
    click.echo("Training complete.")
    click.echo(f"Best AUC: {result.best_score:.4f}")
    click.echo(f"Best params: {result.best_params}")
    click.echo(f"Saved model: {result.best_estimator_path}")


@cli.command()
@click.option(
    "--processed_dir", type=click.Path(path_type=Path), default=PROCESSED_DIR, show_default=True
)
@click.option("--models_dir", type=click.Path(path_type=Path), default=MODELS_DIR, show_default=True)
@click.option("--reports_dir", type=click.Path(path_type=Path), default=REPORTS_DIR, show_default=True)
@click.option(
    "--model_name",
    type=click.Choice(["logistic_regression", "random_forest", "xgboost"], case_sensitive=False),
    default="random_forest",
    show_default=True,
)
def evaluate(processed_dir: Path, models_dir: Path, reports_dir: Path, model_name: str) -> None:
    """Evaluate a saved model and write metrics and plots to the reports directory."""
    metrics = evaluate_model(
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        model_name=model_name,
    )
    click.echo("Evaluation complete. Metrics:")
    for k, v in metrics.items():
        click.echo(f" - {k}: {v:.4f}")


if __name__ == "__main__":
    cli()


