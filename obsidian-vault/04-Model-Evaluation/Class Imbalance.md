# Class Imbalance

## What Is It?

Class imbalance occurs when one class significantly outnumbers the other. In this telecom dataset, non-churners (~73%) vastly outnumber churners (~27%).

## Why It's a Problem

A model that ALWAYS predicts "no churn" achieves **73% accuracy** without learning anything useful. Standard accuracy is misleading.

```
Naive model: predict "no churn" for everyone
Accuracy: 73% ← looks good!
Recall: 0%    ← catches zero churners
Business value: zero
```

## How This Project Handles Class Imbalance

### 1. Stratified Splitting
```python
train_test_split(X, y, stratify=y)
```
Ensures both train and test sets have the same 73/27 ratio. See [[02-Data-Processing/Train-Test Split and Stratification]].

### 2. ROC-AUC Scoring
```python
GridSearchCV(..., scoring="roc_auc")
```
Optimizes for [[04-Model-Evaluation/ROC Curve and AUC|ROC-AUC]] instead of accuracy. AUC evaluates discrimination ability regardless of threshold or class distribution.

### 3. Stratified Cross-Validation
Scikit-learn's `GridSearchCV` uses stratified k-fold by default for classifiers, preserving class ratios in each fold.

## Techniques to Handle Class Imbalance (Full Spectrum)

### Data-Level Techniques

| Technique | How It Works | Pros | Cons |
|-----------|-------------|------|------|
| **Random Oversampling** | Duplicate minority samples | Simple | Risk of overfitting |
| **SMOTE** | Generate synthetic minority samples | Popular, effective | Can create noisy samples |
| **Random Undersampling** | Remove majority samples | Simple, fast | Loses information |
| **Tomek Links** | Remove borderline majority samples | Cleans decision boundary | Minimal effect |

### Algorithm-Level Techniques

| Technique | How It Works |
|-----------|-------------|
| **Class weights** | `class_weight="balanced"` - penalizes misclassifying minority more |
| **Threshold tuning** | Lower threshold from 0.5 to catch more positives |
| **Cost-sensitive learning** | Assign different costs to FP and FN |
| **Anomaly detection** | Treat minority class as "anomalies" |

### Evaluation-Level Techniques

| Metric | Why It Helps |
|--------|-------------|
| ROC-AUC | Threshold-independent, not affected by class ratios |
| F1-Score | Balances precision and recall |
| PR-AUC | More sensitive to minority class performance |
| Recall | Directly measures minority class detection |

## SMOTE (Synthetic Minority Oversampling Technique)

SMOTE generates synthetic samples by interpolating between existing minority samples:

```
Existing sample A: [1.0, 2.0, 3.0]
Existing sample B: [2.0, 4.0, 5.0]
Random factor λ = 0.3

Synthetic sample: A + λ × (B - A)
                = [1.0, 2.0, 3.0] + 0.3 × [1.0, 2.0, 2.0]
                = [1.3, 2.6, 3.6]
```

> **Important:** SMOTE must only be applied to training data, NEVER to test data.

## Improvements Suggested in This Project's README

1. **SMOTE** for oversampling the minority (churn) class
2. **Cost-sensitive learning** with custom loss functions
3. **Threshold tuning** to improve recall from 55.7%
4. **Advanced ensembles** like stacking or blending

## Common Interview Questions

**Q: Your model has 93% accuracy. Is that good?**
A: It depends on the class distribution. With 73% non-churners, a naive model gets 73% accuracy. The 93% is better, but we should look at precision, recall, F1, and AUC to understand true performance on the minority class.

**Q: How would you improve the 55.7% recall in this project?**
A: Five approaches: (1) lower the classification threshold, (2) use `class_weight="balanced"`, (3) apply SMOTE to the training set, (4) engineer better features that capture churn signals, (5) use a cost-sensitive loss function that penalizes FN more heavily.

**Q: When should you NOT use SMOTE?**
A: When the minority class is very noisy, when you have enough data already, when the classes overlap significantly, or when using tree-based models that handle imbalance well with class weights.

---

**Related:** [[04-Model-Evaluation/Precision, Recall, and F1-Score]] | [[04-Model-Evaluation/ROC Curve and AUC]]
