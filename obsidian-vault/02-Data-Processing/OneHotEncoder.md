# OneHotEncoder

## What Is It?

`OneHotEncoder` converts categorical variables into binary (0/1) columns - one column per category. This allows ML models to work with non-numeric data.

## How It's Used in This Project

```python
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
```

## Example

```
Input:                    Output:
┌──────────┐              ┌─────────┬──────────┬──────────┐
│ Contract │              │ Monthly │ One year │ Two year │
├──────────┤              ├─────────┼──────────┼──────────┤
│ Monthly  │     ───>     │    1    │    0     │    0     │
│ Two year │              │    0    │    0     │    1     │
│ One year │              │    0    │    1     │    0     │
└──────────┘              └─────────┴──────────┴──────────┘
```

## Key Parameters

| Parameter | Value in Project | Purpose |
|-----------|-----------------|---------|
| `handle_unknown="ignore"` | Yes | At prediction time, unseen categories become all-zeros instead of erroring |
| `sparse_output=False` | Yes | Returns dense numpy array instead of sparse matrix |
| `drop="first"` | Not used | Would drop first category to avoid multicollinearity |

## handle_unknown="ignore" - Why It Matters

At training time, the encoder learns: `Contract` has values `[Monthly, One year, Two year]`.

If at prediction time a new value appears (e.g., `"Three year"`):
- `handle_unknown="error"` (default): Raises an error
- `handle_unknown="ignore"`: Outputs `[0, 0, 0]` (no category matched)

This makes the model robust to unseen data in production.

## One-Hot vs Other Encoding Methods

| Method | Output | Best For |
|--------|--------|----------|
| **One-Hot** | Binary columns per category | Low-cardinality nominal features |
| Label Encoding | Single column (0, 1, 2...) | Ordinal features or tree models |
| Target Encoding | Category mean of target | High-cardinality features |
| Binary Encoding | Binary representation | High-cardinality features |

## The Dummy Variable Trap

When using One-Hot with linear models ([[03-Machine-Learning/Logistic Regression]]), having all category columns creates **multicollinearity** (one column is perfectly predictable from the others).

Solution: `drop="first"` removes one category column. For `Contract`, you'd have only `One year` and `Two year` columns; `Monthly` is implied when both are 0.

> **Note:** This project doesn't use `drop="first"`. Tree-based models ([[03-Machine-Learning/Random Forest]], [[03-Machine-Learning/XGBoost]]) handle multicollinearity naturally.

## Common Interview Questions

**Q: When would one-hot encoding be a bad choice?**
A: For high-cardinality features (e.g., zip codes with thousands of values) - it creates too many sparse columns. Use target encoding or embeddings instead.

**Q: What's the difference between one-hot encoding and label encoding?**
A: Label encoding assigns integers (0, 1, 2), which implies an ordering that doesn't exist for nominal data. One-hot encoding creates separate binary columns, preserving the fact that categories are unordered.

---

**Related:** [[05-Feature-Engineering/Encoding Categorical Variables]] | [[02-Data-Processing/ColumnTransformer]]
