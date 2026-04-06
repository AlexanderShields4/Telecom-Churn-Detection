# Pytest

## What Is It?

Pytest is Python's most popular testing framework. It discovers and runs test functions, provides rich assertions, and supports fixtures for test setup.

## This Project's Tests

```python
# tests/test_preprocess.py

def test_clean_dataframe_and_preprocessor():
    # Create sample data with known quirks
    df = pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "tenure": [12, 24, None],
        "TotalCharges": ["100.5", " ", "300.0"],   # String with whitespace!
        "Contract": ["Monthly", "One year", "Two year"],
        "Churn": ["Yes", "No", "Yes"],
    })

    # Test cleaning
    cleaned = clean_dataframe(df)
    assert "customerID" not in cleaned.columns        # ID removed
    assert cleaned["Churn"].dtype in [int, np.int64]   # Churn is binary
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])  # Coerced

    # Test preprocessor
    preprocessor = build_preprocessor(cleaned.drop("Churn", axis=1))
    result = preprocessor.fit_transform(cleaned.drop("Churn", axis=1))
    assert result.shape[0] == len(cleaned)             # No rows lost
```

## What's Being Tested

| Assertion | Validates |
|-----------|-----------|
| `customerID` not in columns | ID column removal |
| Churn dtype is int | Target normalization (Yes/No → 0/1) |
| TotalCharges is numeric | String-to-number coercion |
| Row count preserved | Preprocessing doesn't drop rows |
| Output shape matches | Pipeline produces correct dimensions |

## Key Pytest Concepts

### Test Discovery
Pytest finds tests automatically:
- Files matching `test_*.py` or `*_test.py`
- Functions starting with `test_`
- Classes starting with `Test`

### Assertions
```python
assert result == expected           # Equality
assert "key" in dictionary          # Membership
assert value > 0                    # Comparison
assert isinstance(obj, MyClass)     # Type checking

# With message
assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
```

### Fixtures
```python
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({...})

def test_something(sample_dataframe):   # Fixture injected as parameter
    result = process(sample_dataframe)
    assert result is not None
```

### Parametrize
```python
@pytest.mark.parametrize("input,expected", [
    ("Yes", 1), ("No", 0), ("True", 1), ("False", 0),
])
def test_churn_normalization(input, expected):
    assert normalize_churn(input) == expected
```

## Running Tests

```bash
pytest                    # Run all tests
pytest -q                 # Quiet output
pytest -v                 # Verbose output
pytest tests/test_preprocess.py  # Specific file
pytest -k "test_clean"   # Match test name pattern
pytest --tb=short         # Short tracebacks
```

## Common Interview Questions

**Q: What's the difference between unit tests and integration tests?**
A: **Unit tests** test individual functions in isolation (like `test_clean_dataframe`). **Integration tests** test components working together (like the full preprocess → train → evaluate pipeline). This project has unit tests.

**Q: How do you test ML models?**
A: Test data processing (this project does), test model I/O (save/load), test prediction shapes, assert metrics meet minimum thresholds, test edge cases (empty input, single row, all same class).

---

**Related:** [[06-DevOps-and-Tooling/GitHub Actions CI-CD]] | [[06-DevOps-and-Tooling/Linting - Flake8, Black, isort]]
