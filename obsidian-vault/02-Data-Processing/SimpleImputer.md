# SimpleImputer

## What Is It?

`SimpleImputer` fills in missing values (`NaN`) using a simple strategy. It's a scikit-learn transformer that fits on training data and transforms both train and test sets consistently.

## How It's Used in This Project

```python
from sklearn.impute import SimpleImputer

# For numeric columns: fill NaN with median
SimpleImputer(strategy="median")

# For categorical columns: fill NaN with most frequent value
SimpleImputer(strategy="most_frequent")
```

## Imputation Strategies

| Strategy | What It Does | Best For |
|----------|-------------|----------|
| `"mean"` | Replace with column mean | Normal distributions, no outliers |
| `"median"` | Replace with column median | **Skewed data or outliers** (used here) |
| `"most_frequent"` | Replace with mode | **Categorical features** (used here) |
| `"constant"` | Replace with `fill_value` | When missingness itself is meaningful |

## Why Median Over Mean?

`TotalCharges` can have outliers (very high-spending customers). The **median** is robust to outliers:

```
Values: [20, 30, 40, 50, 8000]
Mean:   1628  ← pulled up by outlier
Median: 40    ← represents typical customer
```

## The Importance of Fit/Transform Pattern

```python
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)       # Learns median from training data only
X_train = imputer.transform(X_train)  # Fills NaN with training median
X_test = imputer.transform(X_test)    # Uses SAME training median (not test median)
```

Using the training median for test data prevents [[02-Data-Processing/Scikit-Learn Pipelines|data leakage]].

## When SimpleImputer Isn't Enough

For more sophisticated imputation:
- **`IterativeImputer`** (sklearn) - Models each feature as a function of others (like MICE)
- **`KNNImputer`** (sklearn) - Fills values based on k-nearest neighbors
- **Domain knowledge** - Sometimes missingness has meaning (e.g., missing `TotalCharges` = new customer with no charges)

## Common Interview Questions

**Q: How do you handle missing values?**
A: First, understand *why* they're missing (MCAR, MAR, MNAR). For this project, we use median imputation for numeric features (robust to outliers) and most-frequent imputation for categorical features, applied through a sklearn Pipeline to prevent data leakage.

**Q: What's the difference between MCAR, MAR, and MNAR?**
A:
- **MCAR** (Missing Completely At Random): Missingness unrelated to any data
- **MAR** (Missing At Random): Missingness related to observed data
- **MNAR** (Missing Not At Random): Missingness related to the missing value itself

---

**Related:** [[05-Feature-Engineering/Handling Missing Data]] | [[02-Data-Processing/Scikit-Learn Pipelines]]
