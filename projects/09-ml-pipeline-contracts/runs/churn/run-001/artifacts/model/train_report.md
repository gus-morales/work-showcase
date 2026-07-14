# Training report

## churned_next_30d (classification)

- metric: pr_auc = 0.3918 (goal: maximize, threshold: 0.3)
- reduction funnel: 10 feat / pr_auc=0.3911 -> 9 feat / pr_auc=0.3856 -> 8 feat / pr_auc=0.3792 -> 7 feat / pr_auc=0.3830 -> 6 feat / pr_auc=0.3918
- selected features: plan_tier_basic, tickets_opened_pre_decision, plan_tier_enterprise, avg_usage_score_pre_decision, n_events_pre_decision, avg_logins_pre_decision
