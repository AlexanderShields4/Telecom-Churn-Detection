# Train-Test Split and Stratification

## What Is It?

Splitting data into separate training and test sets to evaluate model performance on unseen data. Stratification ensures the class distribution is preserved in both sets.

## How It's Used in This Project

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 80% train, 20% test
    random_state=42,         # Reproducibility
    stratify=y,              # Preserve class proportions
)
```

## Why Stratify?

With an imbalanced dataset (e.g., 73% non-churn, 27% churn):

### Without stratification (random):
```
Full data:  73% / 27%
Train set:  75% / 25%  ← different distribution
Test set:   68% / 32%  ← different distribution
```

### With stratification:
```
Full data:  73% / 27%
Train set:  73% / 27%  ← same distribution
Test set:   73% / 27%  ← same distribution
```

Stratification ensures the model trains and evaluates on representative data.

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `test_size=0.2` | 20% for test | Standard split ratio |
| `random_state=42` | Fixed seed | Same split every run |
| `stratify=y` | Target column | Preserves class balance |

## Common Split Ratios

| Split | Train | Test | When To Use |
|-------|-------|------|------------|
| 80/20 | 80% | 20% | **Standard** (used here) |
| 70/30 | 70% | 30% | Smaller datasets, need more test data |
| 90/10 | 90% | 10% | Large datasets |

## Train/Validation/Test Split

For more rigorous evaluation:
```
┌─────────────────────────────────────────────┐
│                Full Dataset                  │
├───────────────────────────┬────────┬─────────┤
│      Training (60%)       │Val(20%)│Test(20%)│
└───────────────────────────┴────────┴─────────┘
```

In this project, [[03-Machine-Learning/Cross-Validation]] (5-fold CV within GridSearchCV) serves the validation purpose, so an explicit validation set isn't needed.

## Common Interview Questions

**Q: Why not just use cross-validation instead of a train-test split?**
A: You need a held-out test set that the model NEVER sees during training or hyperparameter tuning. Cross-validation is used during training to select hyperparameters, but the final test set provides an unbiased performance estimate.

**Q: What is data leakage in the context of splitting?**
A: Leakage occurs when test data influences the training process. Examples: fitting a scaler on all data before splitting, or using future data to predict past events. Always split first, then preprocess.

---

**Related:** [[03-Machine-Learning/Cross-Validation]] | [[04-Model-Evaluation/Class Imbalance]]
