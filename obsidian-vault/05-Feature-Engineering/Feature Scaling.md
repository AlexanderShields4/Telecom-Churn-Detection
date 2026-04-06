# Feature Scaling

## Why Scale Features?

Features with different ranges can bias distance-based and gradient-based algorithms. Scaling ensures all features contribute proportionally.

## Example from This Project

| Feature | Raw Range | After StandardScaler |
|---------|-----------|---------------------|
| tenure | 0 - 72 | mean=0, std=1 |
| MonthlyCharges | 18 - 118 | mean=0, std=1 |
| TotalCharges | 0 - 8,684 | mean=0, std=1 |

Without scaling, `TotalCharges` would dominate in [[03-Machine-Learning/Logistic Regression]] because its gradients are much larger.

## Scaling Methods

| Scaler | Formula | Output Range | Best For |
|--------|---------|-------------|----------|
| **StandardScaler** | $(x - \mu) / \sigma$ | ~[-3, 3] | **Normal distributions** (used here) |
| MinMaxScaler | $(x - min) / (max - min)$ | [0, 1] | Neural networks, bounded features |
| RobustScaler | $(x - median) / IQR$ | Varies | Data with outliers |
| MaxAbsScaler | $x / max(\|x\|)$ | [-1, 1] | Sparse data |
| Normalizer | $x / \|\|x\|\|$ | Unit norm | Per-sample normalization |

## Which Models Need Scaling?

| Model | Needs Scaling? | Reason |
|-------|---------------|--------|
| Logistic Regression | **Yes** | Gradient descent + regularization |
| SVM | **Yes** | Distance-based kernel |
| KNN | **Yes** | Distance-based |
| Neural Networks | **Yes** | Gradient descent |
| **Random Forest** | **No** | Tree splits are order-based |
| **XGBoost** | **No** | Tree splits are order-based |
| Naive Bayes | **No** | Probability-based |

> In this project, scaling is applied to all numeric features even though 2 of 3 models don't need it, because the preprocessor is shared across all models.

## Common Interview Questions

**Q: What's the difference between normalization and standardization?**
A: **Standardization** (StandardScaler): centers on mean, scales by std → output has mean=0, std=1. **Normalization** (MinMaxScaler): scales to a fixed range [0,1]. The terms are often used loosely.

**Q: Why must you fit the scaler on training data only?**
A: Fitting on test data leaks information about the test distribution into the model. The scaler must learn mean/std from training data and apply those same values to test data.

---

**Related:** [[02-Data-Processing/StandardScaler]] | [[02-Data-Processing/Scikit-Learn Pipelines]]
