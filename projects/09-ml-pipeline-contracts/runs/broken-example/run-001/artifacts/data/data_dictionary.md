# Data dictionary

One row per `customer_id`. `decision_date` is the point-in-time cutoff every downstream feature must respect.

| Column | Source | Notes |
|---|---|---|
| `customer_id` | base population | primary identifier |
| `decision_date` | base population | point-in-time cutoff |
| `churned_next_30d` | base population (`churned_next_30d`) | target, kind=classification |
