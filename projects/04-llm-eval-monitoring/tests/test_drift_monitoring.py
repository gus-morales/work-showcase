"""Unit tests for the p-chart control-limit and regression-detection logic,
on a small hand-built daily panel with a known, injected step change."""
import pandas as pd
import pytest

from drift_monitoring import compute_control_chart


def _panel(rates, n_tickets=100):
    days = list(range(len(rates)))
    n_acceptable = [round(r * n_tickets) for r in rates]
    return pd.DataFrame({
        "day": days, "n_tickets": [n_tickets] * len(rates),
        "n_acceptable": n_acceptable, "acceptable_rate": rates,
    })


def test_center_line_matches_reference_period_mean():
    df = _panel([0.80] * 10 + [0.50] * 10)
    result = compute_control_chart(df, reference_end_day=9)
    assert result["center"] == pytest.approx(0.80, abs=0.01)


def test_sustained_drop_is_detected_right_after_reference_period():
    df = _panel([0.80] * 10 + [0.50] * 10)
    result = compute_control_chart(df, reference_end_day=9)
    assert result["detected_day"] == 10


def test_clean_panel_with_no_regression_is_not_flagged():
    rates = [0.80, 0.81, 0.79, 0.80, 0.82, 0.79, 0.80, 0.81, 0.79, 0.80,
             0.80, 0.79, 0.81, 0.80, 0.79, 0.81, 0.80, 0.79, 0.80, 0.81]
    df = _panel(rates)
    result = compute_control_chart(df, reference_end_day=9)
    assert result["detected_day"] is None


def test_out_of_control_flags_align_with_the_dropped_days():
    df = _panel([0.80] * 10 + [0.50] * 10)
    result = compute_control_chart(df, reference_end_day=9)
    assert not result["out_of_control"][:10].any()
    assert result["out_of_control"][10:].all()
