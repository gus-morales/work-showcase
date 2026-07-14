---
project:
  name: "Subscription Churn Risk"
  slug: churn
objective: "Flag customers at risk of churning in the next 30 days, around their 90-day mark, so retention outreach can reach them before they leave."
problem: "Retention spend is currently untargeted; the team wants a ranked list of at-risk customers instead of an all-customer campaign."
task_type: classification
targets:
  - name: churned_next_30d
    kind: classification
    raw_field: churned_next_30d
    positive_label: "customer churned within 30 days of the decision date"
metrics:
  - target: churned_next_30d
    name: pr_auc
    goal: maximize
    threshold: 0.30
max_features: 10
data_sources: [usage_events]
status: frozen
---

## Introduction

A subscription business wants to flag, at each customer's 90-day mark, whether they're
likely to churn in the next 30 days, so retention outreach can be targeted instead of
sent to everyone.

## Model Scope

Covers customers reaching their 90-day mark. Does not cover churn prediction earlier in
the lifecycle (day 0-90) or later (post-year-one), both would need their own EDA pass
against a different usage-maturity window.

## Target Proposal

`churned_next_30d`, binary, from `data/churn/customers.csv`. See
`artifacts/eda/target_exploration.json` for the rate-by-month breakdown produced by the
EDA stage.

## Data Sources for Feature Engineering

One event source, `usage_events` (login/feature-usage/support-ticket events), bound in
`bindings.yaml`. Feature ideas in `feature_spec.yaml`: pre-decision usage behavior,
support friction, and plan/region segment.

## Model Evaluation

PR-AUC over ROC-AUC or accuracy: at a ~29% churn rate accuracy is still informative
here (unlike the extreme-imbalance projects elsewhere in this portfolio), but PR-AUC is
kept as the standard for every classification target this framework trains, so the
threshold logic in `stage_io.py` doesn't need a per-run special case.

## Sign-off

Reviewed and frozen for this portfolio example; no external reviewers, synthetic data.
