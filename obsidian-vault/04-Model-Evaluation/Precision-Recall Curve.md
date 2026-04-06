# Precision-Recall Curve

## What Is It?

The Precision-Recall (PR) curve plots [[04-Model-Evaluation/Precision, Recall, and F1-Score|Precision vs Recall]] at every possible classification threshold. It's especially useful for imbalanced datasets.

## How to Read It

```
Precision
1.0 ┤──╲
    │    ╲        Good model
    │     ╲
    │      ╲
0.5 ┤       ╲──────
    │               ╲
    │                 ╲
    │ - - - - - - - - -  Random baseline
    │                     (= class prevalence)
0.0 ┤──────────────────
    0.0       0.5      1.0  Recall
```

The ideal PR curve hugs the **top-right corner** (high precision AND high recall).

## How It's Generated in This Project

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

plt.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("reports/random_forest_pr_curve.png", dpi=200)
```

## PR Curve vs ROC Curve

For this project's churn dataset (~27% positive class):

| Scenario | Use ROC | Use PR |
|----------|---------|--------|
| Moderate imbalance (73/27) | Good choice | Also good |
| Severe imbalance (99/1) | Can be misleading | **Much better** |
| Balanced data (50/50) | Standard choice | Also works |

### Why PR Curve Matters for Imbalanced Data

With 99% negatives:
- ROC FPR = FP / (FP + TN): Even 100 false positives → FPR = 100/9900 = 1% (looks great!)
- PR Precision = TP / (TP + FP): 100 TP + 100 FP → Precision = 50% (reveals the problem!)

## Average Precision (AP)

The area under the PR curve, summarized as a single number:
- 1.0 = perfect
- = prevalence (class proportion) = random baseline

Higher AP means the model maintains high precision even as recall increases.

## Choosing a Threshold from the PR Curve

The PR curve helps select the operating point:

| Business Need | Threshold | Precision | Recall |
|---------------|-----------|-----------|--------|
| "Catch every churner" | Low (0.2) | Lower | ~90% |
| "Only flag confident" | High (0.8) | ~98% | Lower |
| "Balanced" | F1-optimal | Moderate | Moderate |

The **F1-optimal threshold** is where the F1-score is maximized along the curve.

## Common Interview Questions

**Q: When would you use PR curve over ROC curve?**
A: When the positive class is rare (e.g., fraud detection at 0.1% fraud rate). ROC can be overly optimistic because it uses TN in the FPR denominator, making the FPR appear low even with many false positives.

**Q: What's the baseline for a PR curve?**
A: A horizontal line at the prevalence (proportion of positive class). For this churn dataset (~27%), random baseline precision ≈ 0.27.

---

**Related:** [[04-Model-Evaluation/ROC Curve and AUC]] | [[04-Model-Evaluation/Precision, Recall, and F1-Score]]
