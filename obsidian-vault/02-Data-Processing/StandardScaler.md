# StandardScaler

## What Is It?

`StandardScaler` normalizes features by removing the mean and scaling to unit variance. Each feature is transformed to have **mean = 0** and **standard deviation = 1**.

## The Math

$$z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ = original value
- $\mu$ = mean of the feature (from training data)
- $\sigma$ = standard deviation (from training data)
- $z$ = scaled value

## How It's Used in This Project

```python
from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

Applied to numeric columns like `tenure`, `MonthlyCharges`, `TotalCharges`.

## Why Scale Features?

### Without scaling:
- `tenure`: range 0-72 (months)
- `MonthlyCharges`: range 18-118 (dollars)
- `TotalCharges`: range 0-8,600 (dollars)

Models like [[03-Machine-Learning/Logistic Regression]] are **distance/gradient-based**. Without scaling, `TotalCharges` dominates simply because its values are larger, not because it's more important.

### After scaling:
All features have mean ~0 and std ~1. Each feature contributes proportionally to its actual importance.

## Which Models Need Scaling?

| Model | Needs Scaling? | Why |
|-------|---------------|-----|
| [[03-Machine-Learning/Logistic Regression]] | **Yes** | Gradient descent converges faster with scaled features |
| [[03-Machine-Learning/Random Forest]] | **No** | Tree splits are scale-invariant |
| [[03-Machine-Learning/XGBoost]] | **No** | Also tree-based, scale-invariant |

> **Interview insight:** Tree-based models don't need scaling because they make binary decisions at each node (is feature > threshold?). The scale doesn't change the ordering. However, this project applies scaling to all numeric features anyway because the preprocessor is shared across all three models.

## StandardScaler vs Other Scalers

| Scaler | Formula | Best For |
|--------|---------|----------|
| **StandardScaler** | $(x - \mu) / \sigma$ | Normally distributed features |
| MinMaxScaler | $(x - min) / (max - min)$ | Bounded range [0,1] |
| RobustScaler | $(x - median) / IQR$ | Data with outliers |
| MaxAbsScaler | $x / max(\|x\|)$ | Sparse data |

## Common Interview Questions

**Q: When would you NOT use StandardScaler?**
A: When features have many outliers (use RobustScaler instead), when you need a bounded range like [0,1] (use MinMaxScaler), or when using tree-based models that are scale-invariant.

**Q: What happens if you fit the scaler on test data?**
A: Data leakage. The test data's statistics influence the transformation, giving an overly optimistic estimate of model performance. Always fit on training data only.

---

**Related:** [[05-Feature-Engineering/Feature Scaling]] | [[02-Data-Processing/Scikit-Learn Pipelines]]
