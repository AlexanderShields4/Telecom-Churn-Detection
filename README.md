## Churn Prediction in Telecom Industry

This repository implements a complete machine learning pipeline to predict telecom customer churn using classic supervised learning algorithms. It includes data download from Kaggle, preprocessing, model training, evaluation, and CLI entry points.

### Dataset
- Source: `https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets`

### Project Structure
```
.
├── .github/workflows/ci.yml        # CI for lint and tests
├── .gitignore
├── .gitattributes
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── scripts/
│   └── download_data.py           # Kaggle download
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── cli.py                 # CLI entrypoints
│       ├── config.py              # Paths, constants
│       ├── data.py                # Load & preprocess
│       ├── evaluate.py            # Metrics & plots
│       └── models.py              # Model training
├── tests/
│   └── test_preprocess.py
├── data/
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── models/.gitkeep
└── reports/.gitkeep
```

### Project Overview

**What problem are you solving?**
Customer churn is when a telecom subscriber leaves one service provider for another. Predicting churn is essential for telecom companies to reduce customer losses, intervene proactively, and boost revenue.

**What data did you use?**
We utilize the Kaggle Telecom Churn Dataset, combining all provided CSVs for training and evaluation.

**Why is this problem interesting or important?**
Churn prediction enables targeted retention strategies, reducing customer loss and increasing business profitability in highly competitive telecom markets.

### Dataset Description

- **Source:** [Kaggle - Telecom Churn Dataset](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)
- **Size, key features:**
  - ~7,000 samples, 21 features (demographics, account information, call usage, etc.)
  - Binary churn label: `Churn` (Yes/No)
- **Preprocessing steps:**
  - All CSVs merged into a single DataFrame
  - Removed customer ID columns
  - Normalized categorical variables
  - Handled inconsistent/missing values (e.g., `TotalCharges` coerced to numeric)
  - Stratified train/test split
  - Preprocessing pipeline: imputation, scaling of numerics, one-hot-encoding for categoricals

### Modeling Approach

- **Algorithms tried:** Logistic Regression, Random Forest, XGBoost
- **Feature engineering / transformations:** Minimal feature engineering with robust preprocessing pipelines (numeric scaling, categorical one-hot encoding). Hyperparameters tuned via cross-validated grid search optimizing ROC-AUC.

### Results Summary ✅

Best-performing model: **Random Forest**

- **Accuracy:** 0.9340
- **Precision:** 0.9818
- **Recall:** 0.5567
- **F1-score:** 0.7105
- **ROC-AUC:** 0.9297

Detailed classification report:
```
              precision    recall  f1-score   support

           0     0.9297    0.9982    0.9628       570
           1     0.9818    0.5567    0.7105        97

    accuracy                         0.9340       667
   macro avg     0.9558    0.7775    0.8367       667
weighted avg     0.9373    0.9340    0.9261       667
```

Confusion Matrix (embed image):

![Confusion Matrix](reports/random_forest_confusion_matrix.png)

_Summary: Model correctly identified ~56% of churners (positives) and ~100% of non-churners (negatives)._

### Interpretation / Insights

- **What do the results mean?** The model is highly precise when it predicts churn (few false positives) but has moderate recall on churners (some churners are missed). Non-churned customers are classified extremely well.
- **Any surprising findings?** Churners, being a minority, are harder to detect; class imbalance and overlapping patterns likely reduce recall.
- **What could be improved?**
  - Increase recall with class balancing (e.g., SMOTE), cost-sensitive learning, or threshold tuning
  - Explore richer feature engineering, interaction terms, and calibration
  - Try advanced ensembles or stacking and conduct feature importance analysis for business insights
