---
id: DSG-0004
title: "Rebuild the campaign-attribution ETL on the new event schema"
artifact_type: pipeline_change
domain: marketing
impact_level: medium
status: shipped
author: "L. Dubois"
dates:
  proposed: 2026-05-01
  approved: 2026-05-05
  shipped: 2026-05-10
  resolved: null
monitoring:
  ship_check:
    due: 2026-05-17
    done: false
  outcome_check:
    due: 2026-06-09
    done: false
    outcome: null
reviewers: ["A. Fischer", "L. Dubois"]
---

## What changed

Campaign-attribution ETL rebuilt to read from the new marketing-events schema instead of the deprecated one.

## Why

The old schema is being sunset by the events team. Rebuilt the pipeline ahead of that, expected output tables to be identical in shape and values.

## Rollback plan

Old pipeline is still deployed but paused, not deleted. Re-enabling is a scheduler flip back to the previous DAG.

## Monitoring notes

Neither check has been closed out yet.
