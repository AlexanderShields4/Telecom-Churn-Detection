# System Design Questions

## "Design a real-time churn prediction system."

### Architecture

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ Customer  │───>│ Feature      │───>│ Prediction   │───>│ Action   │
│ Events    │    │ Store        │    │ Service      │    │ Engine   │
│ (Kafka)   │    │ (Redis/Feast)│    │ (FastAPI)    │    │ (CRM)    │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘
                       ↑                    ↑
                ┌──────────────┐    ┌──────────────┐
                │ Batch Feature│    │ Model        │
                │ Pipeline     │    │ Registry     │
                │ (Airflow)    │    │ (MLflow)     │
                └──────────────┘    └──────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Event streaming | Kafka | Real-time customer events (calls, billing) |
| Feature store | Feast/Redis | Consistent features for training and serving |
| Model serving | FastAPI + Docker | Low-latency predictions |
| Model registry | MLflow | Version models, track experiments |
| Orchestration | Airflow | Batch retraining pipeline |
| Monitoring | Evidently + Grafana | Data drift, prediction drift |
| Action engine | CRM integration | Trigger retention campaigns |

### Key Design Decisions

1. **Batch vs real-time features**: Some features (tenure, total charges) update daily. Others (recent calls) update in real-time. Use a feature store that supports both.

2. **Model serving latency**: Pre-compute predictions in batch for all customers daily. Real-time predictions only for triggering events (e.g., customer calls to cancel).

3. **Retraining strategy**: Scheduled weekly retraining with automated validation. Retrain immediately if monitoring detects significant data drift.

## "How would you A/B test a new churn model?"

### Approach

```
Incoming customers
       │
  ┌────┴────┐
  │  Router │  (hash customer_id % 100)
  ├─────────┤
  │ 0-49    │ → Model A (current)  → Track churn rate, retention cost
  │ 50-99   │ → Model B (new)      → Track churn rate, retention cost
  └─────────┘
```

### Metrics to Compare
- **Primary**: Reduction in actual churn rate (requires waiting for outcomes)
- **Secondary**: Precision of churn predictions, cost of retention campaigns
- **Guardrail**: Customer satisfaction score, false alarm rate

### Statistical Considerations
- Minimum sample size for statistical power (typically 2-4 weeks)
- Account for seasonality (monthly billing cycles)
- Use the customer as the unit of randomization (not events)

## "How would you handle model retraining?"

### Triggers for Retraining

| Trigger | Detection | Action |
|---------|-----------|--------|
| **Scheduled** | Weekly cron | Retrain on latest data |
| **Data drift** | PSI > 0.2 on key features | Alert + retrain |
| **Performance drop** | AUC drops below 0.85 | Urgent retrain |
| **New features** | Business request | Retrain + evaluate |

### Retraining Pipeline

```
1. Pull latest labeled data (last 6 months)
2. Run preprocessing pipeline
3. Train candidate model
4. Evaluate on holdout set
5. Compare against production model
6. If better → deploy (shadow mode first)
7. Promote to production after validation
```

## "How would you explain model predictions to business stakeholders?"

### SHAP (SHapley Additive exPlanations)

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Per-customer explanation:
# "This customer has high churn risk because:
#  - Month-to-month contract (+0.15)
#  - Low tenure of 3 months (+0.12)
#  - High monthly charges of $95 (+0.08)
#  - No tech support (+0.06)"
```

### Business Dashboard
- Top features driving churn across all customers (global importance)
- Per-customer risk scores with top 3 reasons
- Segment-level churn trends over time

## "What would you change about this project for production?"

| Current | Production |
|---------|-----------|
| CSV/Parquet files | Feature store (Feast) or database |
| Joblib model files | Model registry (MLflow) |
| CLI commands | REST API (FastAPI) + Airflow DAGs |
| Local execution | Docker containers on Kubernetes |
| Manual evaluation | Automated monitoring + alerting |
| Single train/test | Rolling window validation |
| 80/20 split | Time-based split (prevent temporal leakage) |

---

**Related:** [[07-Interview-Questions/Behavioral and Project Questions]] | [[07-Interview-Questions/Technical ML Questions]]
