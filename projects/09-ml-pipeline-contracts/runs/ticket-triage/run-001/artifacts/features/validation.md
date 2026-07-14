# Feature validation

- 10 features, cap was 10
- 2,500 rows, one per `ticket_id`
- max null rate across features: 0.0000
- target leakage check: passed, none of ['will_escalate', 'resolution_hours'] appear in the feature list
