# Random Forest

## What Is It?

Random Forest is an **ensemble learning** method that builds multiple decision trees during training and outputs the **majority vote** (classification) or **mean prediction** (regression) of all trees. It's the best-performing model in this project.

## How It Works

```
                        Full Dataset
                    ┌───────┼───────┐
            Bootstrap 1  Bootstrap 2  Bootstrap 3  ...  Bootstrap N
            (random     (random      (random
             subset)     subset)      subset)
                │           │           │
            Tree 1      Tree 2      Tree 3       ...    Tree N
            (random     (random     (random
             features)   features)   features)
                │           │           │
            Pred: 1     Pred: 0     Pred: 1      ...    Pred: 1
                    └───────┼───────┘
                      Majority Vote
                        Final: 1
```

### Two Sources of Randomness

1. **Bagging (Bootstrap Aggregating)**: Each tree trains on a random sample (with replacement) of the training data
2. **Feature Randomness**: At each split, only a random subset of features is considered

These decorrelate the trees, reducing overfitting.

## How It's Used in This Project

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [200, 400],       # Number of trees
    "max_depth": [None, 10, 20],      # Maximum tree depth
    "min_samples_split": [2, 5],      # Min samples to split a node
}
```

## Key Hyperparameters

| Parameter | Values Tested | Effect |
|-----------|--------------|--------|
| `n_estimators` | 200, 400 | More trees = better performance, diminishing returns, slower |
| `max_depth` | None, 10, 20 | None = fully grown trees. Lower = simpler trees, less overfitting |
| `min_samples_split` | 2, 5 | Higher = more conservative splits, less overfitting |

### Other Important Hyperparameters (not tuned in project)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_features` | `"sqrt"` | Number of features considered per split |
| `min_samples_leaf` | 1 | Minimum samples in a leaf node |
| `max_leaf_nodes` | None | Maximum leaf nodes per tree |
| `class_weight` | None | Can set `"balanced"` for [[04-Model-Evaluation/Class Imbalance]] |

## Why Random Forest Won

1. **Handles non-linearity**: Captures complex interactions between features (e.g., high monthly charges + no contract = high churn)
2. **Robust to outliers**: Tree splits are rank-based, not affected by extreme values
3. **No feature scaling needed**: Decision trees are scale-invariant
4. **Low overfitting risk**: Bagging + feature randomness provides regularization
5. **Minimal feature engineering**: Works well with raw features

## Project Results

| Metric | Score |
|--------|-------|
| Accuracy | 93.4% |
| Precision | 98.2% |
| Recall | 55.7% |
| F1-Score | 71.1% |
| ROC-AUC | 93.0% |

## Feature Importance

Random Forest provides built-in feature importance via `model.feature_importances_`. It measures how much each feature reduces impurity (Gini or entropy) across all trees.

## Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Handles non-linear relationships | Less interpretable than single tree |
| Built-in feature importance | Memory-heavy (stores all trees) |
| Robust to outliers and noise | Slower prediction than logistic regression |
| Rarely overfits with enough trees | Can't extrapolate beyond training range |
| Handles missing values gracefully | Not great for very high-dimensional sparse data |

## Common Interview Questions

**Q: What's the difference between Random Forest and a single Decision Tree?**
A: A single decision tree overfits easily. Random Forest reduces variance by: (1) training each tree on a bootstrap sample, (2) considering random feature subsets at each split, and (3) averaging predictions across hundreds of trees.

**Q: What's the difference between bagging and boosting?**
A: **Bagging** (Random Forest) trains trees independently in parallel on random subsets, then votes. **Boosting** ([[03-Machine-Learning/XGBoost]]) trains trees sequentially, where each tree corrects the errors of the previous one. Bagging reduces variance; boosting reduces bias.

**Q: How does Random Forest handle class imbalance?**
A: Options include `class_weight="balanced"` (weights inversely proportional to class frequencies), adjusting the decision threshold, or using SMOTE for oversampling. This project uses ROC-AUC scoring to optimize for class-imbalanced performance.

---

**Related:** [[03-Machine-Learning/Ensemble Methods]] | [[03-Machine-Learning/XGBoost]] | [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning]]
