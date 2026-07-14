---
id: DSG-0003
title: "Deprecate the legacy relevance-scoring endpoint"
artifact_type: deprecation
domain: search_ranking
impact_level: medium
status: closed
author: "T. Nakamura"
dates:
  proposed: 2026-02-01
  approved: 2026-02-10
  shipped: 2026-02-15
  resolved: 2026-03-18
monitoring:
  ship_check:
    due: 2026-02-22
    done: true
  outcome_check:
    due: 2026-03-17
    done: true
    outcome: iterate
reviewers: ["A. Fischer", "T. Nakamura"]
---

## What changed

The legacy relevance-scoring endpoint (superseded by the ranking service a year ago) was turned off.

## Why

Two internal tools were still calling the legacy endpoint directly instead of the current ranking service. Migrated both before shutdown, expected zero downstream impact.

## Rollback plan

Endpoint code stays in the repo, tagged, for one release cycle. Re-enabling is a one-line config flip if a missed caller turns up.

## Monitoring notes

One caller wasn't migrated in time (an internal reporting script nobody remembered existed) and broke for two days before being fixed. Logged as a process gap for future deprecations: check access logs for callers, not just the known integration list. Keeping the deprecation; treating the miss as something to iterate on for the next one.
