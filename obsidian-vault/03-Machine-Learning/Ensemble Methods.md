# Ensemble Methods

## What Is It?

Ensemble methods combine multiple models to produce a stronger predictor than any individual model. The core idea: a committee of diverse "weak" learners outperforms a single "strong" learner.

## Two Main Families

### 1. Bagging (Bootstrap Aggregating) → [[03-Machine-Learning/Random Forest]]

```
       Dataset
      /   |   \
  Sample Sample Sample    ← Bootstrap samples (with replacement)
    |       |      |
  Tree1  Tree2  Tree3     ← Independent, parallel training
    |       |      |
  Pred1  Pred2  Pred3
      \   |   /
    Majority Vote         ← Aggregation
```

**Goal: Reduce variance (overfitting)**
- Each model sees different data
- Errors cancel out through averaging
- Works best with high-variance models (deep trees)

### 2. Boosting → [[03-Machine-Learning/XGBoost]]

```
  Dataset → Tree1 → Residuals → Tree2 → Residuals → Tree3
                ↓                   ↓                   ↓
           Pred: 2.5          Pred: +0.3          Pred: -0.1

  Final = 2.5 + 0.3 - 0.1 = 2.7
```

**Goal: Reduce bias (underfitting)**
- Each model corrects predecessors' errors
- Sequential, each tree depends on the previous
- Works best with weak learners (shallow trees)

## Comparison Table

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Trees | Deep, fully grown | Shallow (stumps to depth ~7) |
| Focus | Reduce variance | Reduce bias |
| Overfitting risk | Low | Higher (can memorize noise) |
| Example | Random Forest | XGBoost, LightGBM, AdaBoost |
| Key control | Number of trees | Learning rate + number of trees |

## Other Ensemble Techniques

### Stacking
- Train multiple diverse models (e.g., RF + XGBoost + LR)
- Train a "meta-model" on their predictions
- Captures strengths of different model types

### Voting
- **Hard voting**: Majority class prediction
- **Soft voting**: Average predicted probabilities (often better)

### Blending
- Similar to stacking but uses a held-out validation set instead of cross-validation for meta-model training

## Why Ensembles Work: The Wisdom of Crowds

If each model has accuracy p > 0.5 and errors are **independent**, the ensemble's error decreases exponentially with more models.

Example with majority voting of 3 models, each 70% accurate:
```
All 3 correct:  0.7³ = 0.343
2 of 3 correct: 3 × 0.7² × 0.3 = 0.441
Ensemble correct: 0.343 + 0.441 = 0.784  (78.4% > 70%)
```

**Key requirement:** Models must be **diverse** (make different errors). Random Forest achieves this through bootstrap sampling + random feature subsets.

## In This Project

Two ensemble methods are compared:
1. [[03-Machine-Learning/Random Forest]] (bagging) - **Winner** with 93% ROC-AUC
2. [[03-Machine-Learning/XGBoost]] (boosting) - Close performance

Plus [[03-Machine-Learning/Logistic Regression]] as a non-ensemble baseline.

## Common Interview Questions

**Q: Why do ensemble methods generally outperform single models?**
A: By combining diverse models, they reduce variance (bagging) or bias (boosting), and individual model errors tend to cancel out. This is analogous to the "wisdom of crowds" effect.

**Q: What could you stack on top of these three models?**
A: Train all three models, use their predictions as features for a meta-learner (e.g., logistic regression). Use cross-validated predictions to avoid leakage.

---

**Related:** [[03-Machine-Learning/Random Forest]] | [[03-Machine-Learning/XGBoost]]
