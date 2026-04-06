# Encoding Categorical Variables

## Why Encode?

ML models work with numbers. Categorical strings ("Monthly", "One year") must be converted to numeric representations.

## Methods Used in This Project

### One-Hot Encoding (Used)
See [[02-Data-Processing/OneHotEncoder]] for full details.

```
Contract        → Contract_Monthly  Contract_OneYear  Contract_TwoYear
"Monthly"       →       1                0                  0
"One year"      →       0                1                  0
"Two year"      →       0                0                  1
```

## All Encoding Methods

| Method | Output | Ordering | Best For |
|--------|--------|----------|----------|
| **One-Hot** | k binary columns | No | Low cardinality, nominal (this project) |
| **Label** | Single column (0,1,2..) | Implied | Ordinal features, tree models |
| **Ordinal** | Single column with custom order | Yes | Ordinal with known ordering |
| **Target** | Mean of target per category | No | High cardinality |
| **Frequency** | Count/frequency per category | No | High cardinality |
| **Binary** | Binary representation | Partial | Medium cardinality |
| **Embedding** | Dense learned vectors | No | Very high cardinality (NLP) |

## One-Hot vs Label Encoding Decision Tree

```
Is the feature ordinal (has natural order)?
├── Yes → OrdinalEncoder or LabelEncoder
│         (e.g., education: high school < bachelor < master)
└── No (nominal) → Is cardinality low (< ~20)?
    ├── Yes → OneHotEncoder (this project)
    └── No → Target encoding, frequency encoding, or embeddings
```

## The Dummy Variable Trap

With one-hot encoding and linear models, k categories create k columns, but k-1 contain all the information (the kth is perfectly predictable from the rest).

```
# 3 categories, 3 columns (redundant):
Monthly=1, OneYear=0, TwoYear=0
Monthly=0, OneYear=1, TwoYear=0
Monthly=0, OneYear=0, TwoYear=1  ← always = 1 - Monthly - OneYear

# Solution: drop one column (drop="first"):
OneYear=0, TwoYear=0  → implies Monthly
OneYear=1, TwoYear=0
OneYear=0, TwoYear=1
```

This project doesn't use `drop="first"` because tree-based models are unaffected by multicollinearity.

## Common Interview Questions

**Q: Why not just use label encoding for everything?**
A: Label encoding assigns integers (0, 1, 2), implying an order (2 > 1 > 0). For nominal features like "payment method", this creates a false ordering that can mislead models, especially linear ones.

**Q: How do you handle categorical features with many unique values?**
A: Target encoding (replace category with mean of target variable), frequency encoding, hash encoding, or learned embeddings. One-hot encoding creates too many sparse columns for high cardinality.

**Q: What does `handle_unknown="ignore"` do in OneHotEncoder?**
A: At inference time, if a category wasn't seen during training, instead of raising an error, it outputs a zero vector. This makes the model robust to new categories in production data.

---

**Related:** [[02-Data-Processing/OneHotEncoder]] | [[05-Feature-Engineering/Feature Types - Numerical vs Categorical]]
