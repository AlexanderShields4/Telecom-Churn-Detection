# XGBoost

## What Is It?

XGBoost (eXtreme Gradient Boosting) is a **gradient boosting** framework that builds trees sequentially, where each new tree corrects the errors of the ensemble so far. It's known for winning ML competitions and high performance on tabular data.

## How Gradient Boosting Works

```
Step 1:  Initial prediction (e.g., mean)
         ↓
Step 2:  Calculate residuals (errors)
         ↓
Step 3:  Train Tree 1 to predict the residuals
         ↓
Step 4:  Update predictions: pred += learning_rate * Tree1(x)
         ↓
Step 5:  Calculate new residuals
         ↓
Step 6:  Train Tree 2 to predict new residuals
         ↓
         ... repeat for n_estimators trees
```

Each tree is shallow (typically depth 3-7) and fixes a small portion of the remaining error.

## How It's Used in This Project

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    objective="binary:logistic",   # Binary classification
    eval_metric="logloss",         # Evaluation during training
    random_state=42,
    n_estimators=300,              # Number of boosting rounds
    learning_rate=0.1,             # Step size shrinkage
)

param_grid = {
    "max_depth": [3, 5, 7],            # Tree depth (controls complexity)
    "subsample": [0.8, 1.0],           # Row sampling ratio
    "colsample_bytree": [0.8, 1.0],    # Column sampling ratio
}
```

## Key Hyperparameters

### Learning Rate (eta)
- Controls how much each tree contributes
- `0.1` = moderate learning (used here)
- Lower values (0.01) need more trees but generalize better
- **Trade-off**: learning_rate vs n_estimators (lower rate + more trees = better but slower)

### max_depth
- Depth of each individual tree
- `3`: Simple trees, less overfitting (recommended starting point)
- `7`: Complex trees, more expressive but risk overfitting
- Unlike [[03-Machine-Learning/Random Forest]], XGBoost trees should be **shallow**

### Subsample & colsample_bytree
- Introduce randomness similar to Random Forest's bagging
- `subsample=0.8`: Each tree uses 80% of rows
- `colsample_bytree=0.8`: Each tree uses 80% of columns
- Reduces overfitting and speeds up training

### Regularization
| Parameter | Effect |
|-----------|--------|
| `reg_alpha` (L1) | Encourages sparsity in leaf weights |
| `reg_lambda` (L2) | Shrinks leaf weights (default=1) |
| `gamma` | Minimum loss reduction to make a split |

## XGBoost vs Random Forest

| Feature | Random Forest | XGBoost |
|---------|--------------|---------|
| Training | Parallel (independent trees) | Sequential (each tree depends on previous) |
| Strategy | Reduces variance (bagging) | Reduces bias (boosting) |
| Trees | Deep, fully grown | Shallow (depth 3-7) |
| Overfitting risk | Low | Higher (need regularization) |
| Tuning | Fewer hyperparameters | Many hyperparameters |
| Missing values | Not built-in | Handles natively |

## Why XGBoost Didn't Win in This Project

From the README: XGBoost didn't outperform Random Forest because:
1. **Dataset size**: ~7,000 samples - XGBoost shines on larger datasets
2. **Feature set**: The features didn't have enough complex interactions to benefit from sequential correction
3. **Random Forest's simplicity**: With minimal feature engineering, Random Forest was more reliable

> **Interview insight:** XGBoost often excels on larger, more complex datasets. On small, well-structured datasets, simpler models can be equally effective with less tuning.

## XGBoost Special Features

| Feature | Description |
|---------|-------------|
| Built-in missing value handling | Learns optimal direction for missing values at each split |
| Early stopping | Stops training when validation metric stops improving |
| Feature importance | Multiple methods: weight, gain, cover |
| GPU acceleration | `tree_method="gpu_hist"` for large datasets |

## Common Interview Questions

**Q: Explain gradient boosting in simple terms.**
A: We start with a simple prediction. Each subsequent tree learns to predict the errors (residuals) of the ensemble so far. Trees are added one at a time, each slightly improving the overall prediction.

**Q: What's the difference between gradient boosting and AdaBoost?**
A: AdaBoost reweights training samples (misclassified samples get higher weights). Gradient Boosting fits new trees to the negative gradient of the loss function (residuals). Gradient boosting is more general and works with any differentiable loss function.

**Q: How do you prevent overfitting in XGBoost?**
A: (1) Lower learning rate with more trees, (2) limit max_depth, (3) subsample rows and columns, (4) add L1/L2 regularization, (5) use early stopping with a validation set.

**Q: When would you choose XGBoost over Random Forest?**
A: When you have a large dataset, when you need to squeeze out the last bit of performance, when you can afford more hyperparameter tuning, or when you need built-in missing value handling.

---

**Related:** [[03-Machine-Learning/Random Forest]] | [[03-Machine-Learning/Ensemble Methods]] | [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning]]
