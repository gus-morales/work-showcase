---
id: DSG-0001
title: "Move the weekly retention dashboard to a daily refresh"
artifact_type: dashboard_change
domain: product_analytics
impact_level: low
status: closed
author: "M. Alvarez"
dates:
  proposed: 2026-03-01
  approved: 2026-03-02
  shipped: 2026-03-05
  resolved: 2026-04-04
monitoring:
  outcome_check:
    due: 2026-04-04
    done: true
    outcome: keep
reviewers: ["J. Okafor"]
---

## What changed

The retention dashboard's underlying query moved from a weekly to a daily refresh schedule.

## Why

Analysts kept pulling ad hoc numbers between weekly refreshes because the dashboard was stale. Expected this to cut ad hoc query volume against the same tables.

## Monitoring notes

Ad hoc query volume against the retention tables dropped by about a third in the following month. No issues with the daily refresh job. Keeping it as is.
