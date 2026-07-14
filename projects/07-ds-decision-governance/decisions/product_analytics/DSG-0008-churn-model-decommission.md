---
id: DSG-0008
title: "Decommission the 2023 churn-risk model"
artifact_type: deprecation
domain: product_analytics
impact_level: high
status: shipped
author: "J. Okafor"
dates:
  proposed: 2026-04-01
  approved: 2026-04-05
  shipped: 2026-04-10
  resolved: null
monitoring:
  ship_check:
    due: 2026-04-17
    done: false
  outcome_check:
    due: 2026-05-10
    done: false
    outcome: null
reviewers: ["M. Alvarez", "T. Nakamura", "Review Board"]
---

## What changed

The 2023 churn-risk model was taken out of the nightly scoring job; the 2026 model (shipped separately) is now the only one scoring customers.

## Why

Running both models was costing compute for no benefit once the 2026 model's rollout was confirmed stable. Expected no change to any downstream report, since nothing was reading the 2023 model's scores anymore.

## Rollback plan

Old scoring job is disabled, not deleted, and can be re-enabled from the job scheduler in minutes. Score history retained for 90 days.

## Monitoring notes

Not yet checked back on since shipping.
