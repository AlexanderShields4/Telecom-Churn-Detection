# ROC Curve and AUC

## What Is It?

The **ROC (Receiver Operating Characteristic) curve** plots the True Positive Rate vs False Positive Rate at every possible classification threshold. **AUC (Area Under the Curve)** summarizes this into a single number.

## The Axes

- **Y-axis: True Positive Rate (Recall)** = $TP / (TP + FN)$
- **X-axis: False Positive Rate** = $FP / (FP + TN)$

## How to Read It

```
TPR
1.0 ┤         ╭──────────── Perfect model
    │        ╱
    │       ╱  Good model (AUC = 0.93, this project)
    │      ╱
0.5 ┤     ╱
    │    ╱
    │   ╱    ╱ Random guess (AUC = 0.5)
    │  ╱   ╱
    │ ╱  ╱
0.0 ┤╱╱───────────────
    0.0       0.5      1.0  FPR
```

| AUC Score | Interpretation |
|-----------|---------------|
| 1.0 | Perfect classifier |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Fair |
| 0.5 | Random guessing |
| < 0.5 | Worse than random |

## Why AUC Is the Scoring Metric in This Project

```python
GridSearchCV(..., scoring="roc_auc")
```

1. **Threshold-independent**: Evaluates model across ALL thresholds, not just 0.5
2. **Handles class imbalance**: Unlike accuracy, not inflated by majority class. See [[04-Model-Evaluation/Class Imbalance]]
3. **Measures discrimination**: "Can the model rank churners higher than non-churners?"

## How It's Generated in This Project

```python
from sklearn.metrics import roc_curve, roc_auc_score

y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")  # Diagonal baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
```

## Project Result: AUC = 0.93

This means: **93% of the time, the model correctly ranks a random churner higher than a random non-churner.** Excellent discrimination ability.

## ROC vs Precision-Recall Curve

| ROC Curve | Precision-Recall Curve |
|-----------|----------------------|
| FPR vs TPR | Precision vs Recall |
| Less affected by class imbalance | More informative for severe imbalance |
| Baseline: diagonal (0.5 AUC) | Baseline: horizontal at prevalence |
| Standard for balanced/moderate imbalance | Preferred for highly imbalanced (>10:1) |

See [[04-Model-Evaluation/Precision-Recall Curve]] for details.

## Common Interview Questions

**Q: What does AUC = 0.93 mean in practical terms?**
A: If you pick a random churner and a random non-churner, 93% of the time the model assigns a higher churn probability to the actual churner. It measures the model's ability to discriminate between classes.

**Q: Why might you prefer the precision-recall curve over ROC?**
A: With severe class imbalance (e.g., 99% negative), ROC can look good because FPR stays low even with many false positives (denominator TN is huge). PR curve exposes poor precision in these cases.

**Q: Can you have high AUC but poor performance in practice?**
A: Yes. AUC measures ranking ability across all thresholds. In practice, you choose one threshold, and performance at that specific threshold might not match the AUC. Also, AUC doesn't account for the business cost of different types of errors.

---

**Related:** [[04-Model-Evaluation/Precision-Recall Curve]] | [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning]]
