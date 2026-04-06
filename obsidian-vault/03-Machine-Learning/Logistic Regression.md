# Logistic Regression

## What Is It?

Logistic Regression is a **linear classification** model that predicts the probability of a binary outcome using the sigmoid (logistic) function. Despite "regression" in its name, it's used for classification.

## The Math

### Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### The Model
$$P(y=1|x) = \sigma(w^Tx + b) = \frac{1}{1 + e^{-(w^Tx + b)}}$$

Where:
- $w$ = weight vector (one per feature)
- $b$ = bias term
- $x$ = feature vector
- Output: probability between 0 and 1

### Decision Rule
- If $P(y=1|x) \geq 0.5$: predict class 1 (churn)
- If $P(y=1|x) < 0.5$: predict class 0 (no churn)

## How It's Used in This Project

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)

param_grid = {
    "C": [0.1, 1.0, 10.0],       # Regularization strength (inverse)
    "solver": ["lbfgs"],          # Optimization algorithm
}
```

## Key Hyperparameters

### C (Regularization Strength)
- **C is the INVERSE of regularization** - smaller C = stronger regularization
- `C=0.1`: Strong regularization (simpler model, may underfit)
- `C=1.0`: Default balance
- `C=10.0`: Weak regularization (more complex, may overfit)

### Regularization Types
| Type | Penalty | Effect |
|------|---------|--------|
| L1 (Lasso) | $\sum|w_i|$ | Drives some weights to exactly 0 (feature selection) |
| L2 (Ridge) | $\sum w_i^2$ | Shrinks all weights toward 0 (default in sklearn) |
| Elastic Net | Both | Combination of L1 and L2 |

### Solver: LBFGS
Limited-memory Broyden-Fletcher-Goldfarb-Shanno. A quasi-Newton optimization method that approximates the Hessian matrix. Good for small-to-medium datasets with L2 penalty.

## Strengths and Weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Fast to train | Can't capture non-linear relationships |
| Highly interpretable (coefficients = feature importance) | Needs [[02-Data-Processing/StandardScaler\|feature scaling]] |
| Probabilistic output (calibrated probabilities) | Assumes features are roughly independent |
| Good baseline model | Underperforms on complex patterns |

## Performance in This Project

Logistic Regression was the weakest performer because the telecom churn data has **non-linear relationships and feature interactions** that linear models can't capture. It served as a baseline to compare against [[03-Machine-Learning/Random Forest]] and [[03-Machine-Learning/XGBoost]].

## Common Interview Questions

**Q: What's the difference between logistic regression and linear regression?**
A: Linear regression predicts continuous values; logistic regression predicts probabilities for classification by wrapping linear regression in a sigmoid function.

**Q: Why is it called "regression" if it's a classifier?**
A: Historically, it models the log-odds (logit) as a linear function of features. The output is a continuous probability, which is then thresholded for classification.

**Q: How do you interpret logistic regression coefficients?**
A: Each coefficient represents the change in **log-odds** for a one-unit increase in the feature. $e^{w_i}$ gives the odds ratio. Positive coefficient = feature increases probability of churn.

**Q: When would you choose logistic regression over tree-based models?**
A: When interpretability is critical (healthcare, finance), when the relationship is roughly linear, when you have limited data, or as a baseline model.

---

**Related:** [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning]] | [[05-Feature-Engineering/Feature Scaling]]
