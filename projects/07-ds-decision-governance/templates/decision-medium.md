---
id: DSG-XXXX
title: ""
artifact_type: pipeline_change   # dashboard_change | pipeline_change | experiment_rollout | model_launch | metric_definition_change | deprecation
domain: product_analytics        # product_analytics | search_ranking | marketing | customer_support | operations | infrastructure
impact_level: medium
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
reviewers: []  # needs >= 2 (see routing.yaml)
---

## What changed

<!-- One or two sentences. What is this, concretely? -->

## Why

<!-- What prompted this, and what did you expect to happen? -->

## Rollback plan

<!-- Include this section if the change isn't easily reversible. If it
     is (e.g. a dashboard query that can be reverted in one commit),
     it's fine to leave this section out. -->

## Monitoring notes

<!-- Fill in once each check comes due: did it ship as intended
     (ship_check), and did it actually work (outcome_check)? -->
