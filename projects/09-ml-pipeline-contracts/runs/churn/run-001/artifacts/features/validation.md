# Feature validation

- 10 features, cap was 10
- 3,000 rows, one per `customer_id`
- max null rate across features: 0.0000
- target leakage check: passed, none of ['churned_next_30d'] appear in the feature list
