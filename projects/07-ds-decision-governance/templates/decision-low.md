---
id: DSG-XXXX
title: ""
artifact_type: dashboard_change  # dashboard_change | pipeline_change | experiment_rollout | model_launch | metric_definition_change | deprecation
domain: product_analytics        # product_analytics | search_ranking | marketing | customer_support | operations | infrastructure
impact_level: low
status: draft
author: ""
dates:
  proposed: YYYY-MM-DD
  approved: null
  shipped: null
  resolved: null
# Leave monitoring: null until this ships. Low impact means no
# ship_check, but an outcome_check is still required once shipped:
# monitoring:
#   outcome_check:
#     due: <shipped date + ~1 month>
#     done: false
monitoring: null
reviewers: []  # needs >= 1 (see routing.yaml)
---

## What changed

<!-- One or two sentences. What is this, concretely? -->

## Why

<!-- What prompted this, and what did you expect to happen? -->

## Monitoring notes

<!-- Fill in once the outcome_check comes due: what actually happened,
     vs. what you expected in "Why" above. -->
