from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore


@dataclass
class TrainResult:
    model_name: str
    best_estimator_path: Path
    best_params: Dict
    best_score: float


def load_processed_arrays(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = pd.read_parquet(processed_dir / "X_train.parquet").to_numpy()
    X_test = pd.read_parquet(processed_dir / "X_test.parquet").to_numpy()
    y_train = pd.read_parquet(processed_dir / "y_train.parquet")["y"].to_numpy()
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["y"].to_numpy()
    return X_train, X_test, y_train, y_test


def get_model_and_param_grid(model_name: str):
    model_name = model_name.lower()
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=200, n_jobs=None)
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
        }
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ValueError("xgboost is not available. Install xgboost to use this model.")
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
        )
        param_grid = {
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    else:
        raise ValueError(
            "Unknown model_name. Choose from: logistic_regression, random_forest, xgboost"
        )
    return model, param_grid


def train_model(
    processed_dir: Path,
    models_dir: Path,
    model_name: str,
    cv_folds: int = 5,
) -> TrainResult:
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed_arrays(processed_dir)
    model, param_grid = get_model_and_param_grid(model_name)

    grid = GridSearchCV(model, param_grid=param_grid, cv=cv_folds, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    best_path = models_dir / f"{model_name}_best.joblib"
    joblib.dump(best_estimator, best_path)

    return TrainResult(
        model_name=model_name,
        best_estimator_path=best_path,
        best_params=grid.best_params_,
        best_score=float(grid.best_score_),
    )


