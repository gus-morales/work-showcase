# BNPL Delinquency Risk Model

A risk model that flags which buy-now-pay-later loans are likely to go 30+ days past due, and a decision threshold picked from actual business costs instead of a default 0.5 cutoff. Built on synthetic data modeled after a Mexican BNPL lending book, mirroring delinquency-prediction work I currently do in fintech.

**For the full technical walkthrough (modeling, calibration, SHAP, drift monitoring), see the [notebook](notebooks/01_delinquency_risk_model.ipynb).** This README is the short version.

> All data here is synthetically generated. No proprietary data, models, or results from any employer are used or implied.

## The problem

A BNPL lender approves a loan in seconds at checkout. Approve a customer who pays on time, and the lender earns a small merchant fee. Approve one who defaults, and the lender loses most of the loan amount. Getting the approve/decline line right is worth real money.

## What this does

Trains a model to predict delinquency risk at loan approval, then picks the approve/decline threshold that minimizes expected losses given realistic cost assumptions, rather than using a generic 0.5 cutoff.

## Results

Picking the threshold from cost instead of defaulting to 0.5 cut expected portfolio losses by **67%** on held-out data.

| | |
|---|---|
| Model accuracy (AUC) | 0.79 |
| Expected loss reduction vs. a naive 0.5 cutoff | 67% |
| Share of actual delinquent loans caught | 90% |

![Delinquency by month](reports/figures/delinquency_by_month.png)

![Threshold selection](reports/figures/threshold_cost_curve.png)

## Monitoring caught something standard checks would have missed

The model was stress-tested against a simulated economic shock. Standard drift monitoring (checking whether customer profiles have changed) showed nothing unusual. But the actual default rate rose anyway, and the model quietly under-predicted risk during the shock. Catching this took watching the gap between predicted and observed outcomes, not just input drift. Full detail in section 10 of the [notebook](notebooks/01_delinquency_risk_model.ipynb).

![Monitoring](reports/figures/drift_predicted_vs_actual.png)

## Repo layout

- `notebooks/01_delinquency_risk_model.ipynb`: full technical walkthrough, executed with all charts and results inline.
- `src/`: the reproducible pipeline (data generation, features, training, interpretability, monitoring) as standalone scripts.
- `reports/`: generated charts, metrics, and monitoring reports.

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/eda.py
python src/train.py
python src/interpret.py
python src/monitor_drift.py
```

`data/` and `reports/model.pkl` are gitignored; regenerate them by running the scripts above.
