---
id: DSG-0002
title: "Launch attribution model v2 to production scoring"
artifact_type: model_launch
domain: product_analytics
impact_level: high
status: shipped
author: "M. Alvarez"
dates:
  proposed: 2026-06-20
  approved: 2026-06-25
  shipped: 2026-07-01
  resolved: null
monitoring:
  ship_check:
    due: 2026-07-08
    done: true
  outcome_check:
    due: 2026-07-31
    done: false
    outcome: null
reviewers: ["J. Okafor", "R. Singh", "Review Board"]
---

## What changed

Attribution model v2 replaces the rule-based last-touch model for marketing spend attribution.

## Why

The rule-based model was known to overweight the last channel in the path. v2 uses a data-driven multi-touch approach. Expected to shift attributed spend away from paid search toward earlier-funnel channels, without changing the total spend figure.

## Rollback plan

Scoring pipeline can flip back to the v1 rule-based model with a single config change; no data migration involved. If v2's attributed totals diverge from v1's by more than 15% in either direction during the first two weeks, roll back and re-evaluate.

## Monitoring notes

Shipped and scoring live traffic as of 2026-07-01. Ship check confirmed the new model is scoring in production as intended. Outcome check (does the attribution shift match expectations) isn't due until 2026-07-31.
