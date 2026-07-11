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

## About me

Gustavo Morales, Senior Data Scientist, formerly an astrophysicist (PhD, Heidelberg University). 8 years in academic research, 5+ in financial analytics and ML, currently at a BNPL fintech.

[LinkedIn](https://linkedin.com/in/gus-morales) · [GitHub](https://github.com/gus-morales)
