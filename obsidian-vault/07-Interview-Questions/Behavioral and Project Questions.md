# Behavioral and Project Questions

## "Walk me through this project."

> "I built an end-to-end ML pipeline for predicting customer churn in telecom. The pipeline starts with downloading data from Kaggle, then cleans and preprocesses it using scikit-learn Pipelines with separate handling for numeric (median imputation + standard scaling) and categorical features (mode imputation + one-hot encoding).
>
> I trained three models - Logistic Regression as a baseline, Random Forest, and XGBoost - each with hyperparameter tuning via GridSearchCV using 5-fold cross-validation optimizing for ROC-AUC. Random Forest performed best with 93% AUC and 98% precision.
>
> The project is structured as a modular Python package with a CLI interface using Click, automated via Makefile, with CI/CD through GitHub Actions for linting and testing."

## "Why did you choose these three models?"

> "I chose them to represent different approaches:
> - **Logistic Regression** as an interpretable linear baseline
> - **Random Forest** as a bagging ensemble that handles non-linearity well
> - **XGBoost** as a boosting ensemble that often wins competitions
>
> This lets me compare a linear model against two different ensemble strategies. Random Forest won because the dataset was small (~7K samples) and had non-linear feature interactions that logistic regression couldn't capture, while XGBoost didn't have enough data to leverage its sequential learning advantage."

## "What would you do differently?"

> - "Apply **SMOTE** or class weights to address the class imbalance and improve recall from 55.7%"
> - "Add **feature engineering**: interaction features, binning tenure into groups, creating a 'services count' aggregate"
> - "Tune the **classification threshold** based on business cost analysis instead of using default 0.5"
> - "Add **model explainability** with SHAP values to explain predictions to business stakeholders"
> - "Implement **model monitoring** for production deployment (data drift, prediction drift)"

## "How did you handle the class imbalance?"

> "The dataset has roughly 73% non-churners and 27% churners. I used three strategies:
> 1. **Stratified train-test split** to preserve class ratios
> 2. **ROC-AUC scoring** in GridSearchCV instead of accuracy
> 3. **Stratified k-fold** cross-validation (sklearn default for classifiers)
>
> If I were to improve this, I'd add class weights or SMOTE, and tune the decision threshold based on the relative cost of losing a customer vs. the cost of a retention offer."

## "Tell me about a challenge you faced."

> "The `TotalCharges` column contained whitespace strings instead of NaN for new customers. This caused silent type errors downstream. I discovered it during EDA and handled it with `pd.to_numeric(errors='coerce')` to convert whitespace to NaN, which then gets imputed by the pipeline. It reinforced the importance of understanding your data before modeling."

## "How would you deploy this model?"

> "I'd wrap the model and preprocessor in a REST API (FastAPI or Flask), containerize with Docker, and deploy to a cloud service. The preprocessor joblib file ensures consistent feature transformation. I'd add:
> - Input validation and logging
> - A/B testing framework
> - Model monitoring for data drift (using Evidently or similar)
> - Automated retraining pipeline triggered by performance degradation
> - Feature store for consistent feature engineering"

## "How did you ensure reproducibility?"

> "Multiple layers: `random_state=42` for all random operations, pinned dependency versions in `requirements.txt`, stratified splitting, saved preprocessor artifacts, Makefile for standard workflow, and CI/CD that validates the pipeline on every push."

---

**Related:** [[07-Interview-Questions/Technical ML Questions]] | [[00-Project-Overview/Project Architecture]]
