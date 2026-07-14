# Data dictionary

One row per `ticket_id`. `triage_cutoff` is the point-in-time cutoff every downstream feature must respect.

| Column | Source | Notes |
|---|---|---|
| `ticket_id` | base population | primary identifier |
| `triage_cutoff` | base population | point-in-time cutoff |
| `will_escalate` | base population (`will_escalate`) | target, kind=classification |
| `resolution_hours` | base population (`resolution_hours`) | target, kind=regression |
