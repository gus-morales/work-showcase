---
id: DSG-0006
title: "Redefine 'coverage rate' to account for split shifts"
artifact_type: metric_definition_change
domain: operations
impact_level: high
status: draft
author: "K. Nowak"
dates:
  proposed: 2026-07-10
  approved: null
  shipped: null
  resolved: null
monitoring: null
reviewers: []
---

## What changed

Proposal to redefine "coverage rate" so a split shift counts as two partial-coverage periods instead of one full period, matching how staffing actually happens on the floor.

## Why

Coverage rate is used in staffing reports and in at least one automated alert. The current definition overstates coverage on days with a lot of split shifts, which has caused a couple of staffing gaps to go unflagged.

## Rollback plan

Every downstream report and alert that reads "coverage rate" needs to be re-pointed if this changes. Keeping the old definition available under a different name for one quarter so anything not yet migrated doesn't silently break.

## Monitoring notes

Not yet approved.
