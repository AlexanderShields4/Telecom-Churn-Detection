# Scikit-Learn Pipelines

## What Is It?

A `Pipeline` chains multiple preprocessing steps and/or a model into a single object. Each step's output feeds into the next step's input. This prevents data leakage and simplifies code.

## How It's Used in This Project

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numeric pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
```

These pipelines are then combined using a [[02-Data-Processing/ColumnTransformer]].

## Why Pipelines Matter

### Without Pipeline (data leakage risk):
```python
# WRONG: fitting on full data before splitting
scaler.fit(X)          # Learns stats from ALL data (including test)
X_scaled = scaler.transform(X)
X_train, X_test = split(X_scaled)
```

### With Pipeline (correct):
```python
# RIGHT: fit only on training data
pipeline.fit(X_train)             # Learns stats from training only
X_train_out = pipeline.transform(X_train)
X_test_out = pipeline.transform(X_test)  # Uses training stats
```

## Key Concepts

### fit() vs transform() vs fit_transform()

| Method | What It Does | When Used |
|--------|-------------|-----------|
| `fit(X)` | Learns parameters from data (mean, std, categories) | Training data only |
| `transform(X)` | Applies learned parameters to data | Train and test data |
| `fit_transform(X)` | Does both in one step (optimized) | Training data only |

### Data Leakage
When information from the test set influences training. Pipelines prevent this by ensuring `fit()` only sees training data.

> **Interview Key Point:** If you fit a scaler on the entire dataset before splitting, the test set's statistics "leak" into the training process. The model appears to perform better than it actually would on truly unseen data.

### Pipeline Steps
Each step is a tuple `(name, transformer)`:
- `name`: String identifier (used for accessing/setting params)
- `transformer`: Any sklearn-compatible object with `fit()` and `transform()`

### Accessing Pipeline Components
```python
pipeline.named_steps["scaler"]       # Get a specific step
pipeline.set_params(scaler__with_mean=False)  # Set step params (double underscore)
```

## Common Interview Questions

**Q: What is data leakage and how do you prevent it?**
A: Data leakage is when information from the test/validation set influences model training. Prevent it by: (1) splitting before any preprocessing, (2) using sklearn Pipelines to ensure `fit` only touches training data, (3) saving the fitted preprocessor for consistent test-time transforms.

**Q: Can you put a model at the end of a Pipeline?**
A: Yes. `Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])` - calling `pipeline.fit(X, y)` fits everything; `pipeline.predict(X)` transforms and predicts.

---

**Related:** [[02-Data-Processing/ColumnTransformer]] | [[05-Feature-Engineering/Feature Scaling]]
