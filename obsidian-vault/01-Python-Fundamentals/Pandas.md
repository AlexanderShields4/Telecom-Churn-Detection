# Pandas

## What Is It?

Pandas is the core data manipulation library in Python. It provides the `DataFrame` (2D table) and `Series` (1D column) data structures for working with structured data.

## How It's Used in This Project

```python
# Loading multiple CSVs and concatenating
dfs = [pd.read_csv(p) for p in csv_paths]
df = pd.concat(dfs, ignore_index=True)

# Cleaning data
df.drop(columns=id_cols, inplace=True)
df[col] = pd.to_numeric(df[col], errors="coerce")  # Handle non-numeric strings

# Type detection for preprocessing
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Saving to Parquet
df.to_parquet("data/processed/X_train.parquet", index=False)
```

## Key Concepts for Interviews

### DataFrame vs Series
- **DataFrame**: 2D labeled data structure (like a table/spreadsheet)
- **Series**: 1D labeled array (a single column)

### Index
- Every DataFrame has an index (row labels). `ignore_index=True` in `concat` resets it to 0, 1, 2...

### `select_dtypes()`
Used to separate numeric and categorical columns before applying different preprocessing. This is how the project decides which columns go through [[02-Data-Processing/StandardScaler]] vs [[02-Data-Processing/OneHotEncoder]].

### `pd.to_numeric(errors="coerce")`
Converts strings to numbers. `errors="coerce"` turns unparseable values into `NaN` instead of raising an error. Critical for the `TotalCharges` column which has whitespace strings.

### `pd.concat()` vs `pd.merge()`
- **concat**: Stacks DataFrames vertically (rows) or horizontally (columns)
- **merge**: SQL-style joins on keys
- This project uses **concat** to combine multiple CSV files into one DataFrame

### Memory Efficiency
- Parquet files (via [[01-Python-Fundamentals/PyArrow and Parquet]]) are much more memory-efficient than CSV
- `category` dtype can reduce memory for repeated string values

## Common Interview Questions

**Q: How do you handle missing data in pandas?**
A: `df.isnull().sum()` to detect, then `df.fillna()`, `df.dropna()`, or use [[02-Data-Processing/SimpleImputer]] from sklearn.

**Q: What's the difference between `loc` and `iloc`?**
A: `loc` uses label-based indexing, `iloc` uses integer position-based indexing.

**Q: How do you avoid the SettingWithCopyWarning?**
A: Use `.loc[]` for assignments, or explicitly `.copy()` the DataFrame.

**Q: What's the difference between `apply()` and vectorized operations?**
A: Vectorized operations (using numpy under the hood) are much faster. `apply()` iterates row-by-row in Python. Always prefer vectorized operations.

---

**Related:** [[01-Python-Fundamentals/NumPy]] | [[01-Python-Fundamentals/PyArrow and Parquet]]
