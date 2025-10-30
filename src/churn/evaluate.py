from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_processed_arrays(processed_dir: Path):
    X_test = pd.read_parquet(processed_dir / "X_test.parquet").to_numpy()
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["y"].to_numpy()
    return X_test, y_test


def evaluate_model(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    model_name: str,
) -> Dict[str, float]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}_best.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Train first.")

    model = joblib.load(model_path)
    X_test, y_test = load_processed_arrays(processed_dir)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    report_txt = classification_report(y_test, y_pred, digits=4)
    (reports_dir / f"{model_name}_classification_report.txt").write_text(report_txt)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_roc_curve.png", dpi=200)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_pr_curve.png", dpi=200)
    plt.close()

    pd.Series(metrics).to_json(reports_dir / f"{model_name}_metrics.json")

    return metrics


