"""Integration test: run both detection engines end-to-end on the same
synthetic data generate_data.py writes to DuckDB, but through the pure
build_snapshot() function directly, no database or file I/O involved.
Confirms every anomaly deliberately injected in generate_data.py is
still caught after the popmon signal-to-noise tuning."""
from pathlib import Path

from catalog import load_catalog
from generate_data import make_pipeline_runs, make_scoring_log
from snapshot import build_snapshot

BASE = Path(__file__).resolve().parents[1]

# metric -> day range the injected anomaly should surface within
EXPECTED = {
    "pipeline_duration_minutes": range(30, 34),
    "pipeline_success_rate": range(55, 60),
    "data_freshness_hours": range(65, 82),
    "row_count": range(40, 44),
    "monthly_usage_score": range(50, 55),
    "plan_tier": range(60, 65),
    "support_tickets_30d": range(70, 75),
    "predicted_churn_prob": range(20, 81),
}


def test_all_eight_injected_anomalies_are_caught():
    ops_df = make_pipeline_runs()
    scoring_df = make_scoring_log()
    catalog_entries = load_catalog(BASE / "catalog")

    snapshot = build_snapshot(ops_df, scoring_df, catalog_entries)
    events_by_metric = {}
    for event in snapshot["events"]:
        events_by_metric.setdefault(event["metric"], []).append(event["day"])

    for metric, expected_days in EXPECTED.items():
        assert metric in events_by_metric, f"{metric} was never flagged"
        assert any(day in expected_days for day in events_by_metric[metric]), (
            f"{metric} flagged on {events_by_metric[metric]}, expected within {expected_days}"
        )


def test_stable_control_feature_is_never_flagged():
    ops_df = make_pipeline_runs()
    scoring_df = make_scoring_log()
    catalog_entries = load_catalog(BASE / "catalog")

    snapshot = build_snapshot(ops_df, scoring_df, catalog_entries)
    flagged_metrics = {event["metric"] for event in snapshot["events"]}
    assert "tenure_months" not in flagged_metrics
