---
id: DSG-XXXX
title: ""
artifact_type: model_launch      # dashboard_change | pipeline_change | experiment_rollout | model_launch | metric_definition_change | deprecation
domain: product_analytics        # product_analytics | search_ranking | marketing | customer_support | operations | infrastructure
impact_level: high
status: draft
author: ""
dates:
  proposed: YYYY-MM-DD
  approved: null
  shipped: null
  resolved: null
# Leave monitoring: null until this ships, then fill in both checks:
# monitoring:
#   ship_check:
#     due: <shipped date + ~1 week>
#     done: false
#   outcome_check:
#     due: <shipped date + ~1 month>
#     done: false
monitoring: null
reviewers: []  # needs >= 3, including a review-board reviewer (see routing.yaml)
---

## What changed

<!-- One or two sentences. What is this, concretely? -->

## Why

<!-- What prompted this, what did you expect to happen, and what did
     you consider and rule out? -->

## Rollback plan

<!-- Required at this impact level. If this doesn't work as expected,
     what specifically gets undone, and how fast? -->

## Monitoring notes

<!-- Fill in once each check comes due: did it ship as intended
     (ship_check), and did it actually work (outcome_check)? -->
