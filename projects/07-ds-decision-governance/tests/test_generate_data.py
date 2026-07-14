"""Data-generation invariants: schema, ranges, and no unexpected nulls on
a small synthetic sample. Not testing exact values (the generator is
stochastic by design), just the contract the rest of the pipeline relies
on, plus the directional relationships the log is supposed to encode:
higher impact level should mean slower approval, a higher abandonment
rate, a higher monitoring on-time rate, and a lower rollback rate."""
import pytest

import generate_data as gd


@pytest.fixture(scope="module")
def decisions():
    return gd.make_decisions(400)


@pytest.fixture(scope="module")
def approved(decisions):
    return gd.add_approval_outcome(decisions)


@pytest.fixture(scope="module")
def shipped(approved):
    return gd.add_shipping(approved)


@pytest.fixture(scope="module")
def monitored(shipped):
    return gd.add_monitoring_checks(shipped)


@pytest.fixture(scope="module")
def full(monitored):
    return gd.assign_outcome(monitored)


def test_decisions_schema_and_ranges(decisions):
    expected_cols = {"decision_id", "artifact_type", "domain_tag", "impact_level", "proposed_date"}
    assert expected_cols.issubset(decisions.columns)
    assert decisions["decision_id"].is_unique
    assert decisions["artifact_type"].isin(gd.ARTIFACT_TYPES).all()
    assert decisions["domain_tag"].isin(gd.DOMAIN_TAGS).all()
    assert decisions["impact_level"].isin(gd.IMPACT_LEVELS).all()
    assert not decisions.isnull().any().any()


def test_approval_lag_is_nan_only_when_abandoned(approved):
    assert approved.loc[approved["abandoned"], "approval_lag_days"].isna().all()
    assert approved.loc[~approved["abandoned"], "approval_lag_days"].notna().all()
    assert (approved.loc[~approved["abandoned"], "approval_lag_days"] > 0).all()


def test_higher_impact_takes_longer_to_approve(approved):
    lag = approved.dropna(subset=["approval_lag_days"]).groupby("impact_level", observed=True)["approval_lag_days"].mean()
    assert lag["high"] > lag["medium"] > lag["low"]


def test_higher_impact_is_more_likely_to_be_abandoned(approved):
    rate = approved.groupby("impact_level", observed=True)["abandoned"].mean()
    assert rate["high"] > rate["medium"] > rate["low"]


def test_abandoned_decisions_never_ship(shipped):
    assert shipped.loc[shipped["abandoned"], "shipped_date"].isna().all()
    assert shipped.loc[~shipped["abandoned"], "shipped_date"].notna().all()


def test_ship_check_only_required_for_medium_and_high_impact(monitored):
    shipped_rows = monitored[~monitored["abandoned"]]
    required = shipped_rows.groupby("impact_level", observed=True)["ship_check_required"].mean()
    assert required["low"] == 0.0
    assert required["medium"] == 1.0
    assert required["high"] == 1.0


def test_every_shipped_decision_has_a_metric_check_due(monitored):
    shipped_rows = monitored[~monitored["abandoned"]]
    assert shipped_rows["metric_check_due"].notna().all()


def test_higher_impact_has_higher_on_time_rate(monitored):
    rate = monitored.dropna(subset=["metric_check_on_time"]).groupby("impact_level", observed=True)["metric_check_on_time"].mean()
    assert rate["high"] > rate["medium"] > rate["low"]


def test_status_values(full):
    assert full["status"].isin(["abandoned", "reverted", "closed"]).all()
    assert (full.loc[full["abandoned"], "status"] == "abandoned").all()


def test_higher_impact_has_lower_rollback_rate(full):
    resolved = full[full["status"].isin(["closed", "reverted"])]
    rate = resolved.groupby("impact_level", observed=True)["outcome"].apply(lambda s: (s == "rollback").mean())
    assert rate["low"] > rate["medium"] > rate["high"]
