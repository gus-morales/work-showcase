---
project:
  name: "Support Ticket Triage"
  slug: ticket-triage
objective: "Within 2 hours of a ticket opening, predict whether it will escalate and how many hours it will take to resolve, so staffing and routing can react before the ticket ages."
problem: "Escalations and long-running tickets are currently caught only after they've already blown past SLA; the team wants an early-triage signal."
task_type: both
targets:
  - name: will_escalate
    kind: classification
    raw_field: will_escalate
    positive_label: "ticket escalated beyond first-line support"
  - name: resolution_hours
    kind: regression
    raw_field: resolution_hours
metrics:
  - target: will_escalate
    name: pr_auc
    goal: maximize
    threshold: 0.40
  - target: resolution_hours
    name: mae
    goal: minimize
    threshold: 4.5
max_features: 10
data_sources: [agent_activity]
status: frozen
---

## Introduction

A support team wants a two-hour-mark signal for both whether a new ticket is headed
for escalation and roughly how long it will take to resolve, so routing and staffing
can react early instead of after an SLA is already blown.

## Model Scope

Covers the triage decision at the 2-hour mark. Does not cover re-triage later in a
ticket's life or staffing-level optimization, both are downstream of this signal, not
part of it.

## Target Proposal

Two targets from the same base population (`data/ticket_triage/tickets.csv`):
`will_escalate` (binary) and `resolution_hours` (continuous). `task_type: both` is the
point of this run: the same Model Scope, bindings, and stage contracts that handle
project churn's single classification target handle two targets of different kinds
from one base population without any change to the pipeline code.

## Data Sources for Feature Engineering

One event source, `agent_activity` (notes and reassignments), bound in
`bindings.yaml`, filtered to the 2-hour triage window. Feature ideas in
`feature_spec.yaml`: early handling signals and ticket context (category, channel,
customer tier).

## Model Evaluation

`will_escalate` on PR-AUC, same convention as every classification target in this
framework. `resolution_hours` on MAE (mean absolute error in hours), the natural unit
for a staffing conversation ("off by about N hours on average"), goal: minimize.

## Sign-off

Reviewed and frozen for this portfolio example; no external reviewers, synthetic data.
