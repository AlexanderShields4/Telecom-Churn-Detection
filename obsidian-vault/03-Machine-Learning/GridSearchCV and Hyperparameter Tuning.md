# GridSearchCV and Hyperparameter Tuning

## What Is It?

`GridSearchCV` exhaustively searches over a specified grid of hyperparameter values, evaluating each combination using [[03-Machine-Learning/Cross-Validation]]. It returns the combination that produces the best score.

## How It's Used in This Project

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring="roc_auc",       # Optimize for ROC-AUC
    n_jobs=-1,               # Use all CPU cores
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```

## How Grid Search Works

For Random Forest with this param grid:
```python
param_grid = {
    "n_estimators": [200, 400],        # 2 values
    "max_depth": [None, 10, 20],       # 3 values
    "min_samples_split": [2, 5],       # 2 values
}
# Total combinations: 2 x 3 x 2 = 12
# With 5-fold CV: 12 x 5 = 60 model fits
```

```
┌──────────────┬───────────┬─────────────────┬──────────────┐
│ n_estimators │ max_depth │ min_samples_split│ Avg ROC-AUC  │
├──────────────┼───────────┼─────────────────┼──────────────┤
│     200      │   None    │        2        │    0.921     │
│     200      │   None    │        5        │    0.919     │
│     200      │    10     │        2        │    0.928     │
│     ...      │   ...     │       ...       │    ...       │
│     400      │    20     │        5        │    0.930 ← best │
└──────────────┴───────────┴─────────────────┴──────────────┘
```

## Why ROC-AUC Scoring?

The project uses `scoring="roc_auc"` instead of accuracy because of [[04-Model-Evaluation/Class Imbalance]]:

- **Accuracy** with 73/27 imbalance: A model predicting "no churn" always gets 73% accuracy
- **ROC-AUC**: Measures discrimination ability across all thresholds, unaffected by class distribution

## Alternatives to Grid Search

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search** | Try all combinations | Thorough, guaranteed to find best in grid | Exponential time |
| **Random Search** | Random combinations | Faster, often finds good results | May miss optimal |
| **Bayesian Optimization** (Optuna) | Learns from previous results | Efficient, smart exploration | More complex setup |
| **Halving Grid Search** | Progressive elimination | Much faster | Approximate |

### When to Use What

- **Grid Search**: Small param space (< 100 combinations), used in this project
- **Random Search**: Large param space, limited compute budget
- **Bayesian**: Very large space, expensive models, production tuning

## Key GridSearchCV Attributes

| Attribute | Description |
|-----------|-------------|
| `best_estimator_` | The model with best parameters, already fitted |
| `best_params_` | Dict of best parameter values |
| `best_score_` | Mean CV score of best model |
| `cv_results_` | Full results for all parameter combinations |

## Common Interview Questions

**Q: What's the difference between a hyperparameter and a parameter?**
A: **Parameters** are learned during training (weights, biases, tree splits). **Hyperparameters** are set before training and control the learning process (learning rate, max depth, number of trees).

**Q: Why not just evaluate on the test set for each hyperparameter combination?**
A: That would be data leakage - you'd be tuning to perform well on the test set specifically. Cross-validation uses only the training data for hyperparameter selection. The test set is used once at the end for final evaluation.

**Q: How do you handle a very large hyperparameter search space?**
A: Use RandomizedSearchCV (random sampling), Bayesian optimization (Optuna/Hyperopt), or HalvingGridSearchCV (progressive elimination). Also start with a coarse grid, then refine around promising regions.

---

**Related:** [[03-Machine-Learning/Cross-Validation]] | [[04-Model-Evaluation/ROC Curve and AUC]]
