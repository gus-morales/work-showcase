# Work Showcase

[![Tests](https://github.com/gus-morales/work-showcase/actions/workflows/tests.yml/badge.svg)](https://github.com/gus-morales/work-showcase/actions/workflows/tests.yml)

Data science and ML projects demonstrating applied skills in credit risk, customer analytics, and experimentation. This is the kind of work I do as a Senior Data Scientist in fintech, plus a couple of projects outside that scope. All projects use synthetic data generated in-repo, so they're fully shareable. None of it reflects proprietary data or results from any employer.

Each project ships with a pytest test suite (data-generation invariants plus unit tests on the pure computation, e.g. the log-share decomposition, the DiD estimator, the RFM labeling logic) and runs in CI on every push via GitHub Actions, alongside a ruff lint check.

## Projects

| Project | Focus | Status |
|---|---|---|
| [01 - BNPL Delinquency Risk Model](projects/01-delinquency-risk-model) | Credit risk classification, probability calibration, cost-based decisioning, SHAP interpretability, drift/monitoring | Done |
| [02 - Customer LTV & Contribution Analysis](projects/02-ltv-contribution-analysis) | SQL cohort analysis, GMV contribution decomposition, probabilistic CLV (BG/NBD + Gamma-Gamma), early-life predictive modeling | Done |
| [03 - Growth Experimentation & Segmentation](projects/03-growth-experimentation-segmentation) | Power analysis, A/B testing, CUPED, difference-in-differences, RFM/KMeans segmentation, light NLP topic modeling | Done |
| [04 - Evaluating & Monitoring an LLM-Powered Feature](projects/04-llm-eval-monitoring) | LLM-judge validation (Cohen's kappa, bias analysis), A/B testing, statistical process control for output-quality monitoring, cost-based guardrail thresholds | Done |

## Skills and tools featured

**Modeling & ML**

- Classification modeling: logistic regression + gradient-boosted trees (01)
- Gradient boosting regression with feature importance (02)
- Leakage-safe feature pipelines (feature-engine, fit on the training split only) (01)
- Probability calibration (01)
- SHAP interpretability (01)

**Causal inference & experimentation**

- Experiment design and power analysis (03, 04)
- Two-proportion hypothesis testing / A/B testing (03, 04)
- CUPED variance reduction (03)
- Uplift/CATE modeling: T-learner and EconML's CausalForestDML, with Qini-curve validation (03)
- Difference-in-differences with fixed effects and a parallel-trends check (03)
- Wild cluster bootstrap and Honest DiD sensitivity analysis (03)
- DoWhy's causal model/identify/estimate/refute framework (03)

**Customer & growth analytics**

- SQL: window functions, CTEs, cohort joins via DuckDB (02)
- Data contracts and a governed metric layer (02)
- Cohort retention analysis (02)
- Log-share contribution decomposition (02)
- Probabilistic customer lifetime value: BG/NBD + Gamma-Gamma, with calibration/holdout validation (02)
- KMeans clustering for RFM segmentation (03)

**Monitoring & decisioning**

- Drift and calibration monitoring (01)
- Rate-mix shift decomposition (01)
- Statistical process control: p-charts for output-quality monitoring (04)
- Cost-based decision/threshold optimization (01, 04)

**NLP & LLM evaluation**

- LLM-judge validation against human labels: Cohen's kappa, bias decomposition (04)
- Light NLP topic modeling: TF-IDF + NMF (03)

## About me

Gustavo Morales, Senior Data Scientist, formerly an astrophysicist (PhD, Heidelberg University). 8 years in academic research, 5+ in financial analytics and ML, currently at a BNPL fintech.

[LinkedIn](https://linkedin.com/in/gus-morales) · [GitHub](https://github.com/gus-morales)
