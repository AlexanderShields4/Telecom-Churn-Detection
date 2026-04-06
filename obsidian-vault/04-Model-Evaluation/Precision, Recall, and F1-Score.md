# Precision, Recall, and F1-Score

## Definitions

### Precision
> "Of all customers I predicted would churn, how many actually churned?"

$$Precision = \frac{TP}{TP + FP}$$

- **High precision** = few false positives
- Project result: **98.2%** - when the model says "churn", it's almost always right

### Recall (Sensitivity / True Positive Rate)
> "Of all customers who actually churned, how many did I catch?"

$$Recall = \frac{TP}{TP + FN}$$

- **High recall** = few false negatives
- Project result: **55.7%** - the model catches about half of actual churners

### F1-Score
> "A single number that balances precision and recall"

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

- **Harmonic mean** (not arithmetic) - penalizes extreme imbalance
- Project result: **71.1%**

## The Precision-Recall Trade-off

```
Threshold ↑ (stricter)     Threshold ↓ (looser)
  Precision ↑                Precision ↓
  Recall ↓                   Recall ↑
  Fewer predictions          More predictions
  More confident             More aggressive
```

By adjusting the classification threshold (default 0.5):
- **Threshold = 0.3**: Catch more churners (higher recall) but more false alarms (lower precision)
- **Threshold = 0.7**: Only flag very confident predictions (higher precision) but miss more churners (lower recall)

## Which Metric Matters for Churn?

| Scenario | Prioritize | Why |
|----------|-----------|-----|
| **Expensive retention offers** | Precision | Don't waste money on false alarms |
| **High customer lifetime value** | Recall | Don't miss any valuable customers |
| **Balanced approach** | F1-Score | Trade-off between both |
| **This project** | ROC-AUC | Evaluates across all thresholds |

> **This project's trade-off:** 98.2% precision / 55.7% recall means the model is very conservative. In production, you might lower the threshold to catch more churners, accepting more false positives.

## Macro vs Micro vs Weighted Averaging

For multi-class problems (or reporting per-class):

| Average | Formula | When To Use |
|---------|---------|-------------|
| **Macro** | Mean of per-class scores | Equal importance to all classes |
| **Micro** | Global TP, FP, FN | Dominated by majority class |
| **Weighted** | Weighted by class support | Account for class imbalance |

## Common Interview Questions

**Q: When would you optimize for precision over recall?**
A: When false positives are expensive. Examples: spam filtering (don't want to lose real emails), recommending surgery, fraud alerts that freeze accounts.

**Q: When would you optimize for recall over precision?**
A: When false negatives are dangerous. Examples: cancer screening (don't miss a diagnosis), fraud detection (catch all fraud even if some alerts are false), churn prediction (don't lose customers).

**Q: Why use F1 instead of just averaging precision and recall?**
A: F1 uses the harmonic mean, which is closer to the lower value. Example: precision=0.99, recall=0.01 → arithmetic mean=0.50 (misleading), F1=0.02 (correctly shows poor performance).

**Q: How would you improve the recall of this churn model?**
A: (1) Lower the classification threshold below 0.5, (2) use `class_weight="balanced"` in the model, (3) apply SMOTE oversampling, (4) use cost-sensitive learning, (5) engineer features that better capture churn signals.

---

**Related:** [[04-Model-Evaluation/Confusion Matrix]] | [[04-Model-Evaluation/Precision-Recall Curve]] | [[04-Model-Evaluation/Class Imbalance]]
