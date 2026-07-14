# Training report

## will_escalate (classification)

- metric: pr_auc = 0.4353 (goal: maximize, threshold: 0.4)
- reduction funnel: 10 feat / pr_auc=0.4353 -> 9 feat / pr_auc=0.4275 -> 8 feat / pr_auc=0.4043 -> 7 feat / pr_auc=0.4047 -> 6 feat / pr_auc=0.3979
- selected features: early_reassignments, category_general, category_technical, channel_phone, channel_email, early_activity_count, early_notes_added, category_account, channel_chat, customer_tier_basic

## resolution_hours (regression)

- metric: mae = 3.0471 (goal: minimize, threshold: 4.5)
- reduction funnel: 10 feat / mae=3.0471 -> 9 feat / mae=3.0551 -> 8 feat / mae=3.0532 -> 7 feat / mae=3.0754 -> 6 feat / mae=3.0916
- selected features: early_reassignments, category_general, category_technical, channel_phone, channel_email, early_activity_count, early_notes_added, category_account, channel_chat, customer_tier_basic
