---
id: DSG-0007
title: "Add a feature-store read-latency dashboard"
artifact_type: dashboard_change
domain: infrastructure
impact_level: low
status: abandoned
author: "R. Singh"
dates:
  proposed: 2026-01-15
  approved: null
  shipped: null
  resolved: null
monitoring: null
reviewers: []
---

## What changed

Proposed a dashboard tracking p50/p95/p99 read latency on the feature store, broken out by consuming team.

## Why

No visibility into per-team latency, only an aggregate. Wanted to catch a noisy-neighbor problem before it caused a wider outage.

## Monitoring notes

Superseded before approval: the platform team's own latency dashboard (built for an unrelated reason) already covers the same breakdown. Not proceeding with a duplicate.
