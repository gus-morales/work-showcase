"""Tests for the governance analysis outputs, run on the real generated
log: the aggregate directions match generate_data.py's design (higher
impact level -> longer approval lag, lower rollback rate), and the
optional logistic regression returns a sane, in-range result."""
import pandas as pd
import pytest

import generate_data as gd
from governance_analysis import approval_lag_chart, rollback_rate_chart, rollback_logistic_regression

IMPACT_ORDER = ["low", "medium", "high"]


@pytest.fixture(scope="module")
def df():
    data = gd.assign_outcome(gd.add_monitoring_checks(gd.add_shipping(gd.add_approval_outcome(gd.make_decisions(600)))))
    data["impact_level"] = pd.Categorical(data["impact_level"], categories=IMPACT_ORDER, ordered=True)
    return data


def test_approval_lag_increases_with_impact(df, tmp_path, monkeypatch):
    import governance_analysis as ga
    monkeypatch.setattr(ga, "FIG_DIR", tmp_path)
    lag = approval_lag_chart(df, "test")
    assert lag["high"] > lag["medium"] > lag["low"]


def test_rollback_rate_decreases_with_impact(df, tmp_path, monkeypatch):
    import governance_analysis as ga
    monkeypatch.setattr(ga, "FIG_DIR", tmp_path)
    rate = rollback_rate_chart(df, "test")
    assert rate["low"] > rate["medium"] > rate["high"]


def test_logistic_regression_returns_valid_result(df, tmp_path, monkeypatch):
    import governance_analysis as ga
    monkeypatch.setattr(ga, "FIG_DIR", tmp_path)
    result = rollback_logistic_regression(df, "test")
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert result["pr_auc"] >= result["base_rate"]
    assert "impact_level_high" in result["coefficients"]
