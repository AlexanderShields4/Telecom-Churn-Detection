from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _get_feature_names(preprocessor) -> List[str]:
    """Extract human-readable feature names from a fitted ColumnTransformer."""
    names: List[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            out = transformer.get_feature_names_out(columns)
            names.extend(out)
        else:
            names.extend(columns)
    return names


def _shorten_feature_names(names: List[str]) -> List[str]:
    """Clean up sklearn-generated feature names for readable plots."""
    cleaned = []
    for n in names:
        # Remove pipeline prefixes like "cat__onehot__" or "num__"
        for prefix in ("num__", "cat__", "cat__onehot__", "num__scaler__", "num__imputer__"):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break
        cleaned.append(n)
    return cleaned


def explain_model(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    model_name: str,
    top_n: int = 15,
    sample_index: Optional[int] = None,
) -> Dict:
    """Generate comprehensive SHAP explanations with global and local plots.

    Produces:
      - Summary bee-swarm plot (global feature importance with directionality)
      - Bar plot of mean |SHAP| values (global importance ranking)
      - Dependence plots for top drivers (Fiber Optic, Monthly Charges, etc.)
      - Waterfall plot for a single customer (local explanation)
      - Feature importance table as JSON
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    shap_dir = reports_dir / f"{model_name}_shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artifacts ---
    model = joblib.load(models_dir / f"{model_name}_best.joblib")
    preprocessor = joblib.load(processed_dir / "preprocessor.joblib")

    X_test = pd.read_parquet(processed_dir / "X_test.parquet").to_numpy()
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["y"].to_numpy()

    raw_feature_names = _get_feature_names(preprocessor)
    feature_names = _shorten_feature_names(raw_feature_names)

    # Ensure feature name count matches data columns
    if len(feature_names) != X_test.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    feature_names_arr = np.array(feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # --- Compute SHAP values ---
    is_tree = model_name in ("random_forest", "xgboost")
    if is_tree:
        explainer = shap.TreeExplainer(model)
    else:
        # KernelExplainer for logistic regression; use a background sample
        background = shap.sample(X_test_df, min(100, len(X_test_df)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X_test_df)

    # For binary classifiers, shap_values may be:
    #   - a list [class_0, class_1]
    #   - a 3D array (n_samples, n_features, n_classes)
    #   - a 2D array (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_vals = np.array(shap_values[1])
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]  # class 1 = churn
    else:
        shap_vals = shap_values

    # Build an Explanation object for the newer shap API
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        data=X_test_df.values,
        feature_names=feature_names_arr,
    )

    # ================================================================
    # 1) GLOBAL: Bee-swarm summary plot
    # ================================================================
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals,
        X_test_df,
        feature_names=feature_names_arr,
        max_display=top_n,
        show=False,
        plot_size=(12, 8),
    )
    plt.title(f"SHAP Summary (Bee-Swarm) \u2014 {model_name}", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ================================================================
    # 2) GLOBAL: Bar plot of mean |SHAP| values
    # ================================================================
    mean_abs_shap = np.abs(shap_vals).mean(axis=0).ravel()
    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance_df.head(top_n)
    colors = plt.cm.Reds(np.linspace(0.35, 0.85, len(top)))[::-1]
    ax.barh(
        top["feature"].values[::-1],
        top["mean_abs_shap"].values[::-1],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        f"Global Feature Importance (mean |SHAP|) \u2014 {model_name}",
        fontsize=14,
        pad=15,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_global_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ================================================================
    # 3) DEPENDENCE PLOTS for key drivers
    # ================================================================
    # Identify the most relevant feature name variants in the transformed space
    driver_patterns = {
        "Fiber Optic": ["InternetService_Fiber optic", "InternetService_Fiber_optic",
                        "Fiber optic", "Fiber_optic"],
        "Monthly Charges": ["MonthlyCharges", "Monthly Charges", "monthlycharges"],
        "Contract (Month-to-month)": ["Contract_Month-to-month", "Contract_Month_to_month",
                                       "Month-to-month"],
        "Tenure": ["tenure", "Tenure"],
        "Total Charges": ["TotalCharges", "Total Charges", "totalcharges"],
        "Tech Support": ["TechSupport_No", "TechSupport_Yes", "techsupport_No"],
    }

    def _find_feature(patterns: List[str]) -> Optional[str]:
        """Find the first matching feature name (case-insensitive)."""
        lower_map = {f.lower(): f for f in feature_names}
        for pat in patterns:
            if pat.lower() in lower_map:
                return lower_map[pat.lower()]
        return None

    dependence_features_plotted = []
    for label, patterns in driver_patterns.items():
        feat = _find_feature(patterns)
        if feat is None:
            continue
        dependence_features_plotted.append(feat)
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            feat,
            shap_vals,
            X_test_df,
            feature_names=feature_names_arr,
            show=False,
        )
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plt.title(f"SHAP Dependence \u2014 {label}", fontsize=13, pad=12)
        plt.tight_layout()
        plt.savefig(
            shap_dir / f"shap_dependence_{safe_label}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    # Also generate dependence plots for top 2 features by importance if not yet plotted
    for _, row in importance_df.head(2).iterrows():
        feat = row["feature"]
        if feat in dependence_features_plotted:
            continue
        dependence_features_plotted.append(feat)
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            feat,
            shap_vals,
            X_test_df,
            feature_names=feature_names_arr,
            show=False,
        )
        safe = feat.replace(" ", "_").replace("/", "_")
        plt.title(f"SHAP Dependence \u2014 {feat}", fontsize=13, pad=12)
        plt.tight_layout()
        plt.savefig(shap_dir / f"shap_dependence_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ================================================================
    # 4) LOCAL: Waterfall plot for a single customer
    # ================================================================
    if sample_index is None:
        # Pick the highest-risk churner that was correctly predicted
        y_prob = model.predict_proba(X_test)[:, 1]
        churn_mask = y_test == 1
        if churn_mask.any():
            churn_probs = y_prob.copy()
            churn_probs[~churn_mask] = -1  # exclude non-churners
            sample_index = int(np.argmax(churn_probs))
        else:
            sample_index = 0

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation[sample_index], max_display=top_n, show=False)
    plt.title(
        f"Local Explanation \u2014 Customer #{sample_index} (Churn={y_test[sample_index]})",
        fontsize=13,
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(shap_dir / "shap_waterfall_single.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ================================================================
    # 5) LOCAL: Force plot for the same customer (HTML)
    # ================================================================
    force = shap.force_plot(
        (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        shap_vals[sample_index],
        X_test_df.iloc[sample_index],
        feature_names=feature_names_arr,
        matplotlib=False,
    )
    shap.save_html(str(shap_dir / "shap_force_single.html"), force)

    # ================================================================
    # 6) GLOBAL: Force plot for full test set (interactive HTML)
    # ================================================================
    force_all = shap.force_plot(
        (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        shap_vals,
        X_test_df,
        feature_names=feature_names_arr,
        matplotlib=False,
    )
    shap.save_html(str(shap_dir / "shap_force_all.html"), force_all)

    # ================================================================
    # 7) COHORT: Churn vs Non-Churn comparison bar chart
    # ================================================================
    churn_mask = y_test == 1
    if churn_mask.any() and (~churn_mask).any():
        mean_shap_churn = np.abs(shap_vals[churn_mask]).mean(axis=0).ravel()
        mean_shap_no_churn = np.abs(shap_vals[~churn_mask]).mean(axis=0).ravel()

        cohort_df = pd.DataFrame({
            "feature": feature_names,
            "Churned": mean_shap_churn,
            "Retained": mean_shap_no_churn,
        }).set_index("feature")

        # Top features by overall importance
        top_feats = importance_df.head(top_n)["feature"].tolist()
        cohort_df = cohort_df.loc[cohort_df.index.isin(top_feats)]
        cohort_df = cohort_df.loc[top_feats]

        fig, ax = plt.subplots(figsize=(11, 7))
        x = np.arange(len(cohort_df))
        width = 0.35
        ax.bar(x - width / 2, cohort_df["Churned"], width, label="Churned", color="#e74c3c")
        ax.bar(x + width / 2, cohort_df["Retained"], width, label="Retained", color="#2ecc71")
        ax.set_xticks(x)
        ax.set_xticklabels(cohort_df.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(
            f"Feature Impact: Churned vs Retained Customers \u2014 {model_name}",
            fontsize=13,
            pad=15,
        )
        ax.legend(fontsize=11)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_cohort_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()

    # ================================================================
    # 8) Save importance table as JSON + CSV
    # ================================================================
    importance_df.to_csv(shap_dir / "feature_importance_shap.csv", index=False)
    importance_dict = importance_df.head(top_n).to_dict(orient="records")
    with open(shap_dir / "feature_importance_shap.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "top_features": importance_dict,
                "sample_index_explained": sample_index,
                "n_test_samples": len(X_test),
                "plots": [
                    "shap_summary_beeswarm.png",
                    "shap_global_bar.png",
                    "shap_waterfall_single.png",
                    "shap_force_single.html",
                    "shap_force_all.html",
                    "shap_cohort_comparison.png",
                ]
                + [
                    f"shap_dependence_{f.replace(' ', '_').replace('/', '_')}.png"
                    for f in dependence_features_plotted
                ],
            },
            f,
            indent=2,
        )

    return {
        "shap_dir": str(shap_dir),
        "top_features": importance_dict,
        "sample_explained": sample_index,
    }
