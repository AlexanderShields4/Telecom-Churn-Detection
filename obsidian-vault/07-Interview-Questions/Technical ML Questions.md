# Technical ML Questions

## Data & Preprocessing

### "What is data leakage and how did you prevent it?"
> Data leakage is when information from the test set influences training, giving overly optimistic results. I prevented it by:
> 1. Splitting before any preprocessing
> 2. Using sklearn Pipelines - `fit()` only on training data, `transform()` on both
> 3. Saving the fitted preprocessor to ensure identical transforms at inference
>
> See [[02-Data-Processing/Scikit-Learn Pipelines]]

### "Why did you use median imputation instead of mean?"
> Median is robust to outliers. `TotalCharges` has a right-skewed distribution with some very high values that would pull the mean up, making it unrepresentative of typical values.
>
> See [[02-Data-Processing/SimpleImputer]]

### "Why StandardScaler and not MinMaxScaler?"
> StandardScaler assumes roughly normal distributions and centers on mean/std. MinMaxScaler is sensitive to outliers (one extreme value changes the entire range). For this dataset with potential outliers in charges, StandardScaler is more robust.
>
> See [[05-Feature-Engineering/Feature Scaling]]

## Models

### "Explain how Random Forest works."
> Random Forest builds many decision trees, each trained on a bootstrap sample of the data. At each split, only a random subset of features is considered. Final prediction is the majority vote. The randomness decorrelates the trees, reducing variance (overfitting).
>
> See [[03-Machine-Learning/Random Forest]]

### "What's the difference between bagging and boosting?"
> **Bagging** (Random Forest): Trains trees independently in parallel on random subsets, averages predictions. Reduces **variance**.
> **Boosting** (XGBoost): Trains trees sequentially, each correcting the previous ensemble's errors. Reduces **bias**.
>
> See [[03-Machine-Learning/Ensemble Methods]]

### "Explain the bias-variance tradeoff."
> - **Bias**: Error from overly simple assumptions (underfitting). High bias → model misses patterns.
> - **Variance**: Error from sensitivity to training data fluctuations (overfitting). High variance → model memorizes noise.
> - **Tradeoff**: Increasing model complexity reduces bias but increases variance, and vice versa.
> - Random Forest reduces variance through averaging. XGBoost reduces bias through sequential correction.

### "What are the key hyperparameters you tuned?"
> - **Logistic Regression**: `C` (regularization strength) - controls model complexity
> - **Random Forest**: `n_estimators` (tree count), `max_depth` (tree complexity), `min_samples_split` (split constraint)
> - **XGBoost**: `max_depth`, `subsample` (row sampling), `colsample_bytree` (column sampling)
>
> All tuned via GridSearchCV with 5-fold CV optimizing ROC-AUC.
>
> See [[03-Machine-Learning/GridSearchCV and Hyperparameter Tuning]]

### "Why did XGBoost not outperform Random Forest?"
> Three reasons: (1) Small dataset (~7K samples) - XGBoost benefits more from larger data, (2) the feature space didn't have complex sequential patterns that boosting exploits, (3) Random Forest's implicit regularization through bagging was sufficient for this data complexity.

## Evaluation

### "Your recall is 55.7%. How would you improve it?"
> 1. Lower the classification threshold from 0.5 to 0.3
> 2. Use `class_weight="balanced"` to penalize missing churners more
> 3. Apply SMOTE oversampling on training data
> 4. Engineer better features (churn-predictive interactions)
> 5. Use a cost-sensitive loss function
>
> See [[04-Model-Evaluation/Class Imbalance]]

### "Why use ROC-AUC instead of accuracy?"
> With 73% non-churners, a naive "always predict no churn" model gets 73% accuracy. ROC-AUC measures the model's ability to rank churners higher than non-churners across all thresholds, making it robust to class imbalance.
>
> See [[04-Model-Evaluation/ROC Curve and AUC]]

### "Explain the precision-recall tradeoff in business terms."
> **High precision (98.2%)**: When we flag a customer as "will churn", we're right 98% of the time. Very few wasted retention offers.
> **Moderate recall (55.7%)**: We only catch 56% of actual churners. 44% of churners slip through.
> **Business decision**: If retention offers are cheap and losing customers is expensive, we should lower the threshold to catch more churners, accepting more false alarms.

### "What's cross-validation and why 5 folds?"
> K-fold CV splits training data into k folds, trains on k-1, validates on the held-out fold, rotating through all folds. k=5 is the standard - good balance of bias (enough training data per fold) and variance (enough folds for stable estimate).
>
> See [[03-Machine-Learning/Cross-Validation]]

## System Design

### "How would you monitor this model in production?"
> 1. **Data drift**: Monitor input feature distributions (PSI, KS test)
> 2. **Prediction drift**: Monitor prediction distribution over time
> 3. **Performance monitoring**: Track actual churn vs predictions (delayed labels)
> 4. **Alerts**: Trigger retraining when metrics drop below threshold
> 5. **A/B testing**: Compare new model versions against production baseline

### "How would you scale this pipeline?"
> - **Data**: Switch from pandas to Spark/Dask for larger datasets
> - **Training**: Distributed training, use cloud GPU instances for XGBoost
> - **Serving**: Containerize model API, deploy behind load balancer
> - **Pipeline**: Use Airflow/Prefect for orchestration, MLflow for tracking

---

**Related:** [[07-Interview-Questions/Behavioral and Project Questions]] | [[07-Interview-Questions/Python and Coding Questions]]
