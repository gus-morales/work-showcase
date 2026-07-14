# Caught contract violation (real output, not paraphrased)

Stages 1-2 (EDA, Data) ran for real against `runs/churn/run-001`'s Model Scope and
produced a valid `training_data.parquet`. Stage 3 (Feature Engineering) is simulated
here as broken: its declared `feature_cols` include `churned_next_30d`, the target
itself, the single most common way a feature matrix leaks its own label.

```
ContractViolation: [features] target column(s) ['churned_next_30d'] appear in feature_cols, that's target leakage
```

The orchestrator stops here: Training never runs against this feature matrix.
