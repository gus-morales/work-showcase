---
id: DSG-0005
title: "Trial a ticket-volume forecast for next-day staffing"
artifact_type: experiment_rollout
domain: customer_support
impact_level: low
status: reverted
author: "S. Popescu"
dates:
  proposed: 2026-04-01
  approved: 2026-04-02
  shipped: 2026-04-05
  resolved: 2026-04-30
monitoring:
  outcome_check:
    due: 2026-04-28
    done: true
    outcome: rollback
reviewers: ["S. Popescu"]
---

## What changed

A daily ticket-volume forecast was added to the staffing planning sheet, replacing the prior 7-day rolling average.

## Why

The rolling average lagged behind volume spikes around product launches. Expected the forecast to reduce under-staffing on spike days.

## Monitoring notes

Forecast under-predicted two out of three spike days it was tested against, worse than the rolling average it was meant to replace. Reverted to the rolling average; the forecast needs a better handle on launch-day spikes before trying again.
