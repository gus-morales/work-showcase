"""Unit tests for the p-chart control-limit and backlog-detection logic,
on a small hand-built monthly panel with a known, injected step change."""
import pandas as pd
import pytest

from open_loops import compute_control_chart


def _monthly(rates, n_due=100):
    months = list(range(len(rates)))
    n_on_time = [round(r * n_due) for r in rates]
    return pd.DataFrame({
        "month": months, "n_due": [n_due] * len(rates),
        "n_on_time": n_on_time, "on_time_rate": rates,
    })


def test_center_line_matches_reference_period_mean():
    df = _monthly([0.80] * 8 + [0.40] * 8)
    result = compute_control_chart(df, reference_month_end=7)
    assert result["center"] == pytest.approx(0.80, abs=0.01)


def test_sustained_drop_is_detected_right_after_reference_period():
    df = _monthly([0.80] * 8 + [0.40] * 8)
    result = compute_control_chart(df, reference_month_end=7)
    assert result["detected_month"] == 8


def test_clean_panel_with_no_backlog_is_not_flagged():
    rates = [0.80, 0.81, 0.79, 0.80, 0.82, 0.79, 0.80, 0.81, 0.79, 0.80,
             0.80, 0.79, 0.81, 0.80, 0.79, 0.81]
    df = _monthly(rates)
    result = compute_control_chart(df, reference_month_end=7)
    assert result["detected_month"] is None


def test_out_of_control_flags_align_with_the_dropped_months():
    df = _monthly([0.80] * 8 + [0.40] * 8)
    result = compute_control_chart(df, reference_month_end=7)
    assert not result["out_of_control"][:8].any()
    assert result["out_of_control"][8:].all()
