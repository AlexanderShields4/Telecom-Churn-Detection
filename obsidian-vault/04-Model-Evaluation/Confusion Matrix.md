# Confusion Matrix

## What Is It?

A confusion matrix is a table that visualizes model performance by showing the counts of true vs predicted labels for each class.

## The Matrix

```
                    Predicted
                 No Churn    Churn
            ┌────────────┬────────────┐
  No Churn  │    TN       │    FP      │
  (Actual)  │ True Neg    │ False Pos  │
            │ (correct)   │ (Type I)   │
            ├────────────┼────────────┤
   Churn    │    FN       │    TP      │
  (Actual)  │ False Neg   │ True Pos   │
            │ (Type II)   │ (correct)  │
            └────────────┴────────────┘
```

## The Four Outcomes

| Outcome | Meaning | Business Impact (Churn) |
|---------|---------|------------------------|
| **True Positive (TP)** | Predicted churn, actually churned | Correctly identified at-risk customer |
| **True Negative (TN)** | Predicted no churn, actually stayed | Correctly identified loyal customer |
| **False Positive (FP)** | Predicted churn, actually stayed | Wasted retention offer (low cost) |
| **False Negative (FN)** | Predicted no churn, actually churned | **Missed churner** (high cost - lost customer) |

## How It's Generated in This Project

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("reports/random_forest_confusion_matrix.png", dpi=200)
```

## Metrics Derived from the Confusion Matrix

All classification metrics come from TP, TN, FP, FN:

| Metric | Formula | Focus |
|--------|---------|-------|
| **Accuracy** | $(TP + TN) / (TP + TN + FP + FN)$ | Overall correctness |
| **Precision** | $TP / (TP + FP)$ | When you predict positive, are you right? |
| **Recall** | $TP / (TP + FN)$ | Of actual positives, how many did you catch? |
| **F1-Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Balance of precision and recall |
| **Specificity** | $TN / (TN + FP)$ | True negative rate |

See [[04-Model-Evaluation/Precision, Recall, and F1-Score]] for deep dives.

## Reading This Project's Results

Random Forest results:
- **High TN, Low FP**: Very few false alarms → high precision (98.2%)
- **Moderate FN**: Misses ~44% of actual churners → moderate recall (55.7%)

> The model is **conservative**: it only flags a customer as churning when it's very confident. This means few wasted retention offers but some missed churners.

## Common Interview Questions

**Q: What does the confusion matrix tell you that accuracy doesn't?**
A: Accuracy can be misleading with imbalanced data (predicting majority class always gives high accuracy). The confusion matrix shows WHERE the model fails - is it missing positives (high FN) or raising false alarms (high FP)?

**Q: In a churn model, which is worse - FP or FN?**
A: **FN is worse.** A false positive (offering retention to a loyal customer) costs a small incentive. A false negative (missing a churner) costs the entire customer relationship. This argues for optimizing recall, potentially at the cost of some precision.

---

**Related:** [[04-Model-Evaluation/Precision, Recall, and F1-Score]] | [[04-Model-Evaluation/ROC Curve and AUC]]
