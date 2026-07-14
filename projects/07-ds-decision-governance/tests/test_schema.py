"""Unit tests for the record-level validation contract: valid rows from
the real generator pass, and each of the invariants the schema is
supposed to enforce actually catches a broken record."""
import pandas as pd

from schema import validate_dataframe
import generate_data as gd


def _base_record(**overrides):
    record = {
        "decision_id": 1,
        "artifact_type": "model_launch",
        "domain_tag": "product_analytics",
        "impact_level": "medium",
        "status": "closed",
        "proposed_date": "2024-01-01",
        "abandoned": False,
        "approval_lag_days": 5.0,
        "approved_date": "2024-01-06",
        "shipped_date": "2024-01-10",
        "ship_check_required": True,
        "ship_check_due": "2024-01-17",
        "ship_check_on_time": True,
        "metric_check_due": "2024-02-09",
        "metric_check_on_time": True,
        "outcome": "keep",
    }
    record.update(overrides)
    return record


def test_generated_data_has_no_validation_errors():
    df = gd.assign_outcome(gd.add_monitoring_checks(gd.add_shipping(gd.add_approval_outcome(gd.make_decisions(300)))))
    errors = validate_dataframe(df)
    assert errors == []


def test_valid_record_passes():
    df = pd.DataFrame([_base_record()])
    assert validate_dataframe(df) == []


def test_abandoned_record_with_shipped_date_fails():
    df = pd.DataFrame([_base_record(abandoned=True, status="abandoned", outcome=None)])
    errors = validate_dataframe(df)
    assert len(errors) == 1


def test_medium_impact_without_ship_check_fails():
    df = pd.DataFrame([_base_record(ship_check_required=False)])
    errors = validate_dataframe(df)
    assert len(errors) == 1


def test_reverted_status_requires_rollback_outcome():
    df = pd.DataFrame([_base_record(status="reverted", outcome="keep")])
    errors = validate_dataframe(df)
    assert len(errors) == 1


def test_closed_status_cannot_have_rollback_outcome():
    df = pd.DataFrame([_base_record(status="closed", outcome="rollback")])
    errors = validate_dataframe(df)
    assert len(errors) == 1
