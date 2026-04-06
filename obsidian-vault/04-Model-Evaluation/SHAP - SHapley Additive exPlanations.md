# SHAP - SHapley Additive exPlanations

## What Is It?

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain individual predictions. It assigns each feature an importance value (SHAP value) for a particular prediction, based on Shapley values from cooperative game theory.

## How It's Used in This Project

```python
import shap

# TreeExplainer for tree-based models (exact, fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classifiers, use class 1 (churn) values
shap_vals = shap_values[:, :, 1]
```

The project generates 8 types of SHAP outputs via `src/churn/explain.py`:
1. **Bee-swarm summary plot** — global importance with directionality
2. **Bar plot of mean |SHAP|** — global importance ranking
3. **Dependence plots** — per-feature SHAP vs feature value (Fiber Optic, Monthly Charges, Tenure, etc.)
4. **Waterfall plot** — local explanation for a single customer
5. **Force plot (single)** — interactive HTML for one customer
6. **Force plot (all)** — interactive HTML for the full test set
7. **Cohort comparison** — churned vs retained feature impact
8. **Importance table** — CSV/JSON export

## Key Results from This Project

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | Contract (Month-to-month) | 0.0541 |
| 2 | Tenure | 0.0509 |
| 3 | **InternetService (Fiber optic)** | **0.0355** |
| 4 | OnlineSecurity (No) | 0.0316 |
| 5 | TotalCharges | 0.0269 |
| 10 | **MonthlyCharges** | **0.0128** |

**Fiber optic service** and **monthly charges** are confirmed as primary drivers of customer attrition.

## Global vs Local Interpretability

### Global (population-level)
- **Mean |SHAP|**: Average absolute impact across all predictions → feature importance ranking
- **Bee-swarm plot**: Shows distribution of SHAP values for each feature across all samples

### Local (individual prediction)
- **Waterfall plot**: Shows how each feature pushed a specific prediction from the base value
- **Force plot**: Horizontal visualization of feature contributions for one customer

## The Math Behind SHAP

SHAP values are based on **Shapley values** from game theory:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

In plain English: the SHAP value of feature $i$ is the **average marginal contribution** of that feature across all possible orderings of features.

### Properties (Guarantees)
| Property | Meaning |
|----------|---------|
| **Additivity** | SHAP values sum to the prediction minus the base value |
| **Consistency** | If a feature's contribution increases, its SHAP value won't decrease |
| **Local accuracy** | $f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i$ |
| **Missingness** | Features not in the model get SHAP value of 0 |

## SHAP Explainer Types

| Explainer | Models | Speed | Exact? |
|-----------|--------|-------|--------|
| **TreeExplainer** | RF, XGBoost, LightGBM | Fast | Yes |
| KernelExplainer | Any model | Slow | Approximate |
| LinearExplainer | Linear models | Fast | Yes |
| DeepExplainer | Neural networks | Medium | Approximate |

This project uses **TreeExplainer** for Random Forest (exact, polynomial time).

## SHAP Dependence Plot — Reading Guide

```
SHAP value ↑  (pushes toward churn)
    |   • •
    |  • • •
  0 |--------•-------
    |       • •
    |      • • •
SHAP value ↓  (pushes away from churn)
    └──────────────→  Feature value
```

- Each dot is one customer
- X-axis: the feature's actual value
- Y-axis: SHAP impact on prediction
- Color: interaction feature (automatically selected)

## Common Interview Questions

**Q: What's the difference between SHAP and traditional feature importance?**
A: Traditional importance (e.g., Gini importance in Random Forest) measures how much a feature reduces impurity globally. SHAP provides per-prediction explanations, shows directionality (positive/negative impact), and has theoretical guarantees (additivity, consistency). SHAP is also model-agnostic.

**Q: How do you explain a model's prediction to a business stakeholder?**
A: Using a SHAP waterfall plot: "This customer has an 89.9% churn probability. The biggest drivers are: low tenure (+13%), month-to-month contract (+6%), fiber optic service (+5%), and no online security (+4%). These features pushed the prediction up from a baseline of 26.5%."

**Q: Why is Fiber Optic a churn driver? Isn't it a premium service?**
A: Fiber optic customers likely pay more (higher MonthlyCharges) and may have higher expectations. If service quality doesn't match the premium price, dissatisfaction grows. The SHAP dependence plot confirms the interaction between fiber optic and monthly charges amplifies churn risk.

**Q: What's the difference between SHAP and LIME?**
A: LIME (Local Interpretable Model-agnostic Explanations) fits a local linear model around each prediction by perturbing input. SHAP is based on game theory with stronger theoretical guarantees. SHAP values are additive and consistent; LIME approximations can be unstable across runs.

---

**Related:** [[04-Model-Evaluation/Class Imbalance]] | [[03-Machine-Learning/Random Forest]] | [[04-Model-Evaluation/Precision, Recall, and F1-Score]]
