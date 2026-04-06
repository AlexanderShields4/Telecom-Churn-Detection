# Python and Coding Questions

## Python Fundamentals

### "What Python features does this project use?"
> - **Dataclasses** for structured return types (`TrainResult`, `ProcessedPaths`)
> - **Type hints** throughout for self-documenting code
> - **Pathlib** for cross-platform path handling
> - **Decorators** via Click for CLI commands
> - **List comprehensions** for data loading (`[pd.read_csv(p) for p in paths]`)
> - **Context managers** (`with` statements) for file handling

### "Explain Python decorators."
> A decorator wraps a function to modify its behavior. In this project, Click uses decorators to convert functions into CLI commands:
> ```python
> @click.command()          # Wraps function as CLI command
> @click.option("--name")   # Adds CLI option
> def my_command(name):
>     ...
> ```
> Internally: `my_command = click.command()(click.option("--name")(my_command))`

### "What are `*args` and `**kwargs`?"
> `*args`: Collects positional arguments into a tuple.
> `**kwargs`: Collects keyword arguments into a dict.
> Used for flexible function signatures, common in sklearn's `set_params(**params)`.

### "Mutable default argument trap?"
> ```python
> # BAD: list is shared across calls
> def f(items=[]):
>     items.append(1)
>     return items
>
> # GOOD: use None sentinel
> def f(items=None):
>     items = items or []
>     items.append(1)
>     return items
> ```
> This is why dataclasses use `field(default_factory=list)`.

## Data Processing

### "How would you optimize pandas operations for large datasets?"
> 1. Use `read_csv` with `dtype` and `usecols` to limit memory
> 2. Use categorical dtype for low-cardinality string columns
> 3. Use vectorized operations instead of `apply()`
> 4. Process in chunks with `chunksize` parameter
> 5. Switch to Polars or Dask for truly large data

### "Write a function to detect and handle outliers."
> ```python
> def remove_outliers_iqr(df, column, factor=1.5):
>     Q1 = df[column].quantile(0.25)
>     Q3 = df[column].quantile(0.75)
>     IQR = Q3 - Q1
>     lower = Q1 - factor * IQR
>     upper = Q3 + factor * IQR
>     return df[(df[column] >= lower) & (df[column] <= upper)]
> ```

## Software Engineering

### "Why structure the project as a Python package?"
> - **Importability**: `from churn.data import clean_dataframe` works from anywhere
> - **Testability**: Functions can be imported and tested in isolation
> - **CLI**: `python -m churn.cli` runs the package as a module
> - **Separation of concerns**: Each module has a clear responsibility

### "How do you handle configuration?"
> Centralized in `config.py` with constants:
> ```python
> RANDOM_STATE = 42
> TEST_SIZE = 0.2
> TARGET_COLUMN = "Churn"
> ```
> Paths are derived from `PROJECT_ROOT` using `pathlib.Path`. This avoids magic numbers and strings scattered throughout the code.

### "What's the difference between `__init__.py` and `__main__.py`?"
> - `__init__.py`: Makes a directory a Python package. Runs on `import churn`.
> - `__main__.py`: Runs when the package is executed: `python -m churn`.

## Coding Challenges

### "Implement a simple k-fold cross-validation from scratch."
> ```python
> def k_fold_cv(X, y, model, k=5):
>     fold_size = len(X) // k
>     scores = []
>     for i in range(k):
>         val_start = i * fold_size
>         val_end = val_start + fold_size
>         X_val = X[val_start:val_end]
>         y_val = y[val_start:val_end]
>         X_train = np.concatenate([X[:val_start], X[val_end:]])
>         y_train = np.concatenate([y[:val_start], y[val_end:]])
>         model.fit(X_train, y_train)
>         scores.append(model.score(X_val, y_val))
>     return np.mean(scores)
> ```

### "Implement precision and recall from scratch."
> ```python
> def precision_recall(y_true, y_pred):
>     tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
>     fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
>     fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
>     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
>     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
>     return precision, recall
> ```

---

**Related:** [[07-Interview-Questions/Technical ML Questions]] | [[01-Python-Fundamentals/Dataclasses]]
