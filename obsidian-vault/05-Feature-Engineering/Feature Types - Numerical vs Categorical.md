# Feature Types - Numerical vs Categorical

## Overview

ML models require numeric input. Understanding feature types determines which preprocessing to apply.

## Feature Types in This Project

### Numerical Features
Features with continuous or discrete numeric values:

| Feature | Type | Example Values |
|---------|------|---------------|
| tenure | Discrete | 1, 12, 45, 72 (months) |
| MonthlyCharges | Continuous | 18.25, 56.95, 118.75 |
| TotalCharges | Continuous | 0, 1397.47, 8684.80 |
| SeniorCitizen | Binary | 0, 1 |

**Preprocessing**: [[02-Data-Processing/SimpleImputer]] (median) → [[02-Data-Processing/StandardScaler]]

### Categorical Features
Features with discrete, non-numeric categories:

| Feature | Cardinality | Example Values |
|---------|-------------|---------------|
| gender | 2 | Male, Female |
| Contract | 3 | Month-to-month, One year, Two year |
| PaymentMethod | 4 | Electronic check, Mailed check, Bank transfer, Credit card |
| InternetService | 3 | DSL, Fiber optic, No |

**Preprocessing**: [[02-Data-Processing/SimpleImputer]] (most_frequent) → [[02-Data-Processing/OneHotEncoder]]

## How the Project Detects Feature Types

```python
# Automatic detection based on pandas dtype
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
```

These lists are passed to the [[02-Data-Processing/ColumnTransformer]] to route each feature to the right pipeline.

## Special Cases

### Binary Numeric (SeniorCitizen)
Already 0/1 - could be treated as numeric or categorical. This project treats it as numeric (no encoding needed).

### Ordinal vs Nominal Categorical
- **Nominal**: No natural order (gender, PaymentMethod) → One-Hot Encoding
- **Ordinal**: Natural order (Contract: month < 1yr < 2yr) → Could use label encoding

This project treats all categoricals as nominal with one-hot encoding. For ordinal features, `OrdinalEncoder` preserves the ordering.

## Common Interview Questions

**Q: How do you decide if a feature is categorical or numerical?**
A: Consider the semantics: (1) Can you meaningfully average it? → Numeric. (2) Does it represent groups/categories? → Categorical. (3) Edge case: zip codes are numeric but should be treated as categorical.

**Q: What about high-cardinality categorical features (e.g., 1000+ categories)?**
A: One-hot encoding creates too many columns. Alternatives: target encoding, frequency encoding, embeddings, or feature hashing.

---

**Related:** [[02-Data-Processing/ColumnTransformer]] | [[05-Feature-Engineering/Encoding Categorical Variables]]
