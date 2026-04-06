# Joblib

## What Is It?

Joblib is a Python library for efficient serialization of Python objects (especially NumPy arrays) and lightweight parallelism. In ML, it's the standard way to save and load trained models.

## How It's Used in This Project

```python
import joblib

# Saving a trained model
joblib.dump(best_model, "models/random_forest_best.joblib")

# Saving the preprocessor pipeline
joblib.dump(preprocessor, "data/processed/preprocessor.joblib")

# Loading for evaluation
model = joblib.load("models/random_forest_best.joblib")
preprocessor = joblib.load("data/processed/preprocessor.joblib")
```

## Why Joblib Over Pickle?

| Feature | pickle | joblib |
|---------|--------|--------|
| NumPy arrays | Slow (serializes element by element) | Fast (memory-maps large arrays) |
| Large objects | All in memory | Can use disk-backed storage |
| Compression | Manual | Built-in (`compress` parameter) |
| ML ecosystem | Generic | Standard in scikit-learn |

## Key Concepts

### Serialization (Persistence)
Converting a Python object (model, pipeline, scaler) to bytes that can be saved to disk and reconstructed later.

### Why Save the Preprocessor?
The preprocessor learns statistics from training data:
- [[02-Data-Processing/StandardScaler]]: learns mean and std
- [[02-Data-Processing/OneHotEncoder]]: learns category mappings
- [[02-Data-Processing/SimpleImputer]]: learns median/mode values

These **must** be the same at training and inference time. Saving the fitted preprocessor prevents **data leakage** and ensures consistency.

### Parallelism in GridSearchCV
```python
GridSearchCV(..., n_jobs=-1)  # Uses joblib internally for parallel CV folds
```
`n_jobs=-1` means "use all CPU cores". Joblib manages the parallel execution backend.

## Security Warning

> Never load a joblib/pickle file from an untrusted source. Deserialization can execute arbitrary code. This is a common interview topic.

---

**Related:** [[02-Data-Processing/Scikit-Learn Pipelines]] | [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning]]
