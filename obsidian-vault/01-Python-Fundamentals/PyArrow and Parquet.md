# PyArrow and Parquet

## What Is Parquet?

Apache Parquet is a **columnar storage format** designed for efficient data processing. Unlike CSV (row-based), Parquet stores data column-by-column, enabling better compression and faster analytical queries.

## How It's Used in This Project

```python
# Saving processed data
X_train_df.to_parquet("data/processed/X_train.parquet", index=False)
y_train_df.to_parquet("data/processed/y_train.parquet", index=False)

# Loading processed data
X_train = pd.read_parquet("data/processed/X_train.parquet")
```

PyArrow (`pyarrow==17.0.0`) is the engine that Pandas uses to read/write Parquet files.

## Parquet vs CSV

| Feature | CSV | Parquet |
|---------|-----|---------|
| Format | Row-based text | Columnar binary |
| Types | Everything is text | Preserves int, float, string, datetime |
| Size | Large | 2-10x smaller (compression) |
| Read speed | Slow (parse text) | Fast (binary, column pruning) |
| Schema | None (inferred) | Embedded schema |
| Human readable | Yes | No |

## Why Columnar Storage Matters

```
CSV (row-based):         Parquet (columnar):
Name, Age, City          Name: [Alice, Bob, Charlie]
Alice, 30, NYC           Age:  [30, 25, 35]
Bob, 25, LA              City: [NYC, LA, CHI]
Charlie, 35, CHI
```

Benefits of columnar:
- **Column pruning**: Read only the columns you need
- **Better compression**: Similar values in a column compress well
- **Vectorized processing**: Operate on entire columns at once

## Interview Relevance

**Q: Why did you use Parquet instead of CSV for intermediate data?**
A: Three reasons: (1) Parquet preserves column types so we don't re-infer dtypes on every load, (2) it's smaller on disk due to columnar compression, and (3) it's faster to read, especially when you only need a subset of columns.

**Q: What's the difference between PyArrow and fastparquet?**
A: Both are Parquet engines for Python. PyArrow (Apache Arrow) is more widely used, faster, and supports more features. It's the default engine in modern Pandas.

---

**Related:** [[01-Python-Fundamentals/Pandas]] | [[00-Project-Overview/Project Architecture]]
