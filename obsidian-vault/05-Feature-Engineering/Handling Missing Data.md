# Handling Missing Data

## Types of Missingness

| Type | Meaning | Example | Implication |
|------|---------|---------|-------------|
| **MCAR** | Missing Completely At Random | Random survey non-response | Safe to impute or drop |
| **MAR** | Missing At Random | Income missing more for younger people | Impute using observed features |
| **MNAR** | Missing Not At Random | High-income people hide income | Missingness is informative |

## How This Project Handles Missing Data

### TotalCharges Problem
Some `TotalCharges` values are whitespace strings (new customers with no charges):
```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# Whitespace → NaN, then imputed with median
```

### Imputation Strategy
```python
# Numeric: median (robust to outliers)
SimpleImputer(strategy="median")

# Categorical: most frequent value (mode)
SimpleImputer(strategy="most_frequent")
```

See [[02-Data-Processing/SimpleImputer]] for details.

## Common Strategies

| Strategy | How | When To Use |
|----------|-----|-------------|
| **Drop rows** | `df.dropna()` | Few missing rows, MCAR |
| **Drop columns** | `df.drop(col)` | Column mostly missing (>50%) |
| **Mean/Median** | `SimpleImputer` | Numeric, quick baseline |
| **Mode** | `SimpleImputer(strategy="most_frequent")` | Categorical |
| **KNN Imputer** | Based on similar samples | Features are correlated |
| **Iterative** | Model each feature | Complex relationships |
| **Missingness indicator** | Add `is_missing` column | Missingness is informative (MNAR) |

## Best Practices

1. **Investigate before imputing** - understand WHY data is missing
2. **Fit on train only** - imputation statistics from training data applied to test
3. **Consider adding a missingness indicator** for features where missing = meaningful
4. **Pipeline integration** - use sklearn imputers inside Pipelines to prevent leakage

## Common Interview Questions

**Q: How do you handle missing data?**
A: First, analyze the pattern and proportion of missingness. For this project, numeric features use median imputation (robust to outliers) and categorical features use mode imputation, both within a sklearn Pipeline to prevent data leakage.

**Q: When is dropping missing data acceptable?**
A: When missingness is MCAR and the proportion is very small (<5%). Dropping loses information and can introduce bias if data isn't MCAR.

---

**Related:** [[02-Data-Processing/SimpleImputer]] | [[02-Data-Processing/Scikit-Learn Pipelines]]
