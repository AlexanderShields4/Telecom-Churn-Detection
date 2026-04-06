# Dataclasses

## What Is It?

Python's `@dataclass` decorator (stdlib, Python 3.7+) automatically generates `__init__`, `__repr__`, `__eq__`, and other boilerplate for classes that primarily hold data.

## How It's Used in This Project

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ProcessedPaths:
    """Paths to all processed data artifacts."""
    X_train: Path
    X_test: Path
    y_train: Path
    y_test: Path
    preprocessor: Path

@dataclass
class TrainResult:
    """Result of a training run."""
    model_name: str
    model_path: Path
    best_params: dict
    best_score: float
```

## Why Use Dataclasses?

### Without dataclass:
```python
class TrainResult:
    def __init__(self, model_name, model_path, best_params, best_score):
        self.model_name = model_name
        self.model_path = model_path
        self.best_params = best_params
        self.best_score = best_score

    def __repr__(self):
        return f"TrainResult(model_name={self.model_name}, ...)"
```

### With dataclass:
```python
@dataclass
class TrainResult:
    model_name: str
    model_path: Path
    best_params: dict
    best_score: float
# __init__, __repr__, __eq__ all auto-generated
```

## Key Features

| Feature | Code | Effect |
|---------|------|--------|
| Frozen (immutable) | `@dataclass(frozen=True)` | Raises error on attribute assignment |
| Default values | `field: int = 0` | Sets default in generated `__init__` |
| Factory defaults | `field: list = field(default_factory=list)` | Avoids mutable default bug |
| Post-init | `def __post_init__(self)` | Runs after `__init__` for validation/computation |

## Dataclass vs NamedTuple vs Dict

| | Dict | NamedTuple | Dataclass |
|--|------|------------|-----------|
| Mutable | Yes | No | Yes (unless frozen) |
| Type hints | No | Yes | Yes |
| Auto methods | No | Yes | Yes |
| Inheritance | N/A | Limited | Full |
| Memory | Higher | Lower | Medium |

## Interview Relevance

Using dataclasses shows you understand:
- **Clean code** - Structured return types instead of raw tuples/dicts
- **Type annotations** - Self-documenting interfaces
- **Python idioms** - Using modern Python features appropriately

---

**Related:** [[01-Python-Fundamentals/Joblib]]
