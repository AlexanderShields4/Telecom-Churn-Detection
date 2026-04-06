# ColumnTransformer

## What Is It?

`ColumnTransformer` applies different transformations to different subsets of columns in a DataFrame. Essential when your data has mixed types (numeric + categorical).

## How It's Used in This Project

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),    # StandardScaler for numbers
        ("cat", categorical_pipeline, categorical_cols),  # OneHotEncoder for strings
    ]
)
```

Where:
- `numeric_cols` = columns with dtype `number` (e.g., tenure, MonthlyCharges)
- `categorical_cols` = columns with dtype `object`/`category` (e.g., Contract, PaymentMethod)

## How It Works

```
Input DataFrame:
┌─────────┬──────────┬──────────┬──────────┐
│ tenure  │ Monthly  │ Contract │ Payment  │
│ (float) │ (float)  │ (str)    │ (str)    │
├─────────┼──────────┼──────────┼──────────┤
│ 12      │ 29.85    │ Monthly  │ Credit   │
│ 45      │ 56.95    │ Two year │ Bank     │
└─────────┴──────────┴──────────┴──────────┘
        │                    │
   numeric_pipeline     categorical_pipeline
   (Impute → Scale)    (Impute → OneHot)
        │                    │
        ▼                    ▼
┌─────────┬──────────┬───┬───┬───┬───┬───┐
│ tenure  │ Monthly  │ M │ 1y│ 2y│ Cr│ Bk│
│ (scaled)│ (scaled) │   │   │   │   │   │
├─────────┼──────────┼───┼───┼───┼───┼───┤
│ -1.2    │ -0.8     │ 1 │ 0 │ 0 │ 1 │ 0 │
│  0.9    │  0.3     │ 0 │ 0 │ 1 │ 0 │ 1 │
└─────────┴──────────┴───┴───┴───┴───┴───┘
```

## Key Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `remainder` | `"drop"` (default) | Drops columns not in any transformer |
| `remainder` | `"passthrough"` | Keeps untransformed columns as-is |
| `sparse_threshold` | `0.3` (default) | Output sparse matrix if enough sparse output |
| `n_jobs` | `-1` | Parallelize transformer fitting |

## Column Selection Methods

```python
# By column names (used in this project)
ColumnTransformer([("num", scaler, ["tenure", "MonthlyCharges"])])

# By dtype using make_column_selector
from sklearn.compose import make_column_selector
ColumnTransformer([
    ("num", scaler, make_column_selector(dtype_include="number")),
    ("cat", encoder, make_column_selector(dtype_include="object")),
])
```

## Common Interview Questions

**Q: How does your project handle mixed feature types?**
A: We use ColumnTransformer to route numeric columns through a median imputer + StandardScaler pipeline, and categorical columns through a most-frequent imputer + OneHotEncoder pipeline. The outputs are concatenated into a single feature matrix.

**Q: What happens to columns not specified in any transformer?**
A: By default (`remainder="drop"`), they're dropped. You can set `remainder="passthrough"` to keep them unchanged.

---

**Related:** [[02-Data-Processing/Scikit-Learn Pipelines]] | [[02-Data-Processing/StandardScaler]] | [[02-Data-Processing/OneHotEncoder]]
