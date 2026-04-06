# NumPy

## What Is It?

NumPy (Numerical Python) is the foundational library for numerical computing in Python. It provides the `ndarray` - a fast, memory-efficient multi-dimensional array that underpins Pandas, scikit-learn, and virtually all Python ML libraries.

## How It's Used in This Project

NumPy operates mostly behind the scenes:

```python
# Scikit-learn transformers output NumPy arrays
X_train_processed = preprocessor.fit_transform(X_train)  # Returns ndarray

# Model predictions are NumPy arrays
y_pred = model.predict(X_test)       # ndarray of 0s and 1s
y_proba = model.predict_proba(X_test)[:, 1]  # ndarray of probabilities
```

## Key Concepts for Interviews

### ndarray vs Python List
| Feature | Python List | NumPy Array |
|---------|------------|-------------|
| Speed | Slow (Python loop) | Fast (C-optimized) |
| Memory | Each element is a Python object | Contiguous memory block |
| Operations | No vectorization | Vectorized math |
| Types | Mixed types | Homogeneous type |

### Vectorization
```python
# Slow Python loop
result = [x * 2 for x in my_list]

# Fast NumPy vectorized operation
result = my_array * 2  # Operates on entire array at once
```

### Broadcasting
NumPy automatically expands arrays of different shapes during arithmetic:
```python
# Scalar broadcast: (1000,) * scalar
scaled = array * 2.0

# Shape broadcast: (1000, 5) - (1, 5)
centered = data - data.mean(axis=0)  # Subtract column means
```

This is exactly what [[02-Data-Processing/StandardScaler]] does: `(X - mean) / std`

### Key Functions Used by sklearn

| Function | Purpose |
|----------|---------|
| `np.mean()`, `np.std()` | Used internally by StandardScaler |
| `np.median()` | Used by SimpleImputer with `strategy="median"` |
| `np.concatenate()` | Used by ColumnTransformer to merge transformed columns |
| `np.argmax()` | Used internally for classification predictions |

### dtype (Data Type)
Every array has a single dtype (`float64`, `int32`, etc.). This is why arrays are fast - no per-element type checking.

## Common Interview Questions

**Q: Why is NumPy faster than Python lists?**
A: Three reasons: (1) contiguous memory layout for cache efficiency, (2) operations implemented in C, (3) vectorization avoids Python interpreter overhead per element.

**Q: What is broadcasting?**
A: NumPy's mechanism for performing operations on arrays of different shapes by virtually expanding the smaller array to match the larger one, without copying data.

**Q: What's the difference between a copy and a view?**
A: A **view** shares memory with the original (slicing creates views). A **copy** is independent (`array.copy()`). Modifying a view modifies the original.

---

**Related:** [[01-Python-Fundamentals/Pandas]] | [[02-Data-Processing/StandardScaler]]
