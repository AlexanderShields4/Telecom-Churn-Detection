# Cross-Validation

## What Is It?

Cross-validation is a technique for evaluating model performance by splitting the training data into multiple folds. The model is trained on some folds and validated on the held-out fold, rotating through all folds.

## K-Fold Cross-Validation (Used in Project: k=5)

```
Fold 1: [VAL] [Train] [Train] [Train] [Train]  → Score: 0.91
Fold 2: [Train] [VAL] [Train] [Train] [Train]  → Score: 0.93
Fold 3: [Train] [Train] [VAL] [Train] [Train]  → Score: 0.90
Fold 4: [Train] [Train] [Train] [VAL] [Train]  → Score: 0.92
Fold 5: [Train] [Train] [Train] [Train] [VAL]  → Score: 0.94

Average Score: 0.92 ± 0.015
```

Each data point is in the validation set exactly once. The final score is the **average** across all folds.

## How It's Used in This Project

```python
GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,              # 5-fold cross-validation
    scoring="roc_auc",
)
```

For each hyperparameter combination, 5-fold CV produces an average ROC-AUC score. The combination with the highest average wins.

## Why Use Cross-Validation?

### Single train/validation split problems:
- Results depend heavily on which data ends up in validation
- Small validation set → high variance in performance estimate
- Wastes data (can't use validation data for training)

### Cross-validation solves all three:
- Every data point gets evaluated
- Average over 5 folds reduces variance
- Each fold uses 80% of data for training

## Types of Cross-Validation

| Type | Description | Use Case |
|------|-------------|----------|
| **K-Fold** | Split into k equal folds | Standard (used here, k=5) |
| **Stratified K-Fold** | K-Fold preserving class distribution | **Imbalanced data** (sklearn default for classifiers) |
| **Leave-One-Out** | k = number of samples | Very small datasets |
| **Time Series Split** | Forward-looking folds only | Temporal data (no future leakage) |
| **Repeated K-Fold** | Run K-Fold multiple times | More robust estimate |
| **Group K-Fold** | Keeps groups together | Prevents data leakage from grouped data |

> **Important:** Scikit-learn's `GridSearchCV` automatically uses **Stratified K-Fold** for classifiers, which preserves the churn/non-churn ratio in each fold. This is critical for [[04-Model-Evaluation/Class Imbalance]].

## Choosing K

| K | Training Size | Bias | Variance | Compute |
|---|--------------|------|----------|---------|
| 3 | 67% | Higher | Lower | Fast |
| **5** | **80%** | **Moderate** | **Moderate** | **Standard** |
| 10 | 90% | Lower | Higher | Slow |
| N (LOO) | N-1 | Lowest | Highest | Very slow |

**k=5** is the standard choice (used in this project) - good balance of bias, variance, and compute.

## Common Interview Questions

**Q: What's the bias-variance tradeoff in choosing k?**
A: Higher k = more training data per fold = lower bias but higher variance (each fold's validation set is smaller). Lower k = less training data = higher bias but lower variance.

**Q: When would you NOT use standard k-fold CV?**
A: With time series data (use TimeSeriesSplit), grouped data (use GroupKFold), or extremely large datasets (a single validation split may suffice).

---

**Related:** [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning]] | [[02-Data-Processing/Train-Test Split and Stratification]]
