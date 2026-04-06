# Dataset and Business Context

## The Business Problem

**Customer churn** = when a customer stops using a company's service. In telecom, acquiring a new customer costs **5-25x more** than retaining an existing one. Predicting churn allows companies to:

- Proactively offer retention deals to at-risk customers
- Identify root causes of dissatisfaction
- Optimize marketing spend (target likely churners)
- Reduce revenue loss

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | [Kaggle - Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets) |
| Samples | ~7,000 customers |
| Features | 21 |
| Target | `Churn` (binary: Yes/No) |
| Class Balance | Imbalanced (more non-churners than churners) |

## Feature Categories

### Demographics
- Gender, SeniorCitizen, Partner, Dependents

### Account Information
- Tenure (months with company), Contract type, PaymentMethod, PaperlessBilling
- MonthlyCharges, TotalCharges

### Services
- PhoneService, MultipleLines, InternetService
- OnlineSecurity, OnlineBackup, DeviceProtection
- TechSupport, StreamingTV, StreamingMovies

## Data Quality Issues Handled

1. **`TotalCharges` as string** - Some values are whitespace strings, not numbers. Coerced to numeric with `pd.to_numeric(..., errors="coerce")`
2. **Customer ID columns** - Removed (not predictive, would cause overfitting)
3. **Inconsistent churn labels** - Normalized Yes/No/True/False/1/0 to binary 0/1
4. **Missing values** - Handled via [[02-Data-Processing/SimpleImputer]] (median for numeric, most-frequent for categorical)

## Why This Matters for Interviews

This is a **classic binary classification problem** that touches on:
- [[04-Model-Evaluation/Class Imbalance]] - Real-world datasets are rarely balanced
- [[05-Feature-Engineering/Feature Types - Numerical vs Categorical]] - Mixed feature types require different preprocessing
- Business impact of [[04-Model-Evaluation/Precision, Recall, and F1-Score]] trade-offs (false positives vs false negatives)

> **Interview Insight:** A false negative (missing a churner) costs the company a lost customer. A false positive (flagging a loyal customer) only costs a retention offer. This asymmetry should guide your choice of evaluation metric.

---

**Related:** [[00-Project-Overview/Project Architecture]] | [[04-Model-Evaluation/Class Imbalance]]
