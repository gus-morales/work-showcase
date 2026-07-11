"""Data-generation invariants for all four synthetic datasets."""
import pytest

import generate_data as gd


# --- 1. Experiment users (A/B test) -----------------------------------

@pytest.fixture(scope="module")
def experiment_users():
    return gd.make_experiment_users(n=2000)


def test_experiment_users_schema(experiment_users):
    expected = {"user_id", "arm", "revenue_pre_30d_usd", "converted_post_14d", "revenue_post_14d_usd"}
    assert expected.issubset(experiment_users.columns)
    assert experiment_users["user_id"].is_unique
    assert experiment_users["arm"].isin(["control", "treatment"]).all()
    assert experiment_users["converted_post_14d"].isin([0, 1]).all()
    assert experiment_users["revenue_pre_30d_usd"].ge(0).all()
    assert not experiment_users.isnull().any().any()


def test_treatment_arm_has_higher_conversion(experiment_users):
    rates = experiment_users.groupby("arm")["converted_post_14d"].mean()
    assert rates["treatment"] > rates["control"]


def test_pre_and_post_revenue_are_correlated(experiment_users):
    # This is what makes CUPED useful downstream; if it collapses to ~0
    # the variance-reduction story breaks.
    corr = experiment_users["revenue_pre_30d_usd"].corr(experiment_users["revenue_post_14d_usd"])
    assert corr > 0.2


# --- 2. Regional rollout panel (diff-in-diff) --------------------------

@pytest.fixture(scope="module")
def rollout():
    return gd.make_regional_rollout(n_regions=8, n_days=40, rollout_day=20)


def test_rollout_schema_and_ranges(rollout):
    expected = {"region_id", "group", "day", "post_rollout", "n_customers", "n_on_time", "on_time_rate"}
    assert expected.issubset(rollout.columns)
    assert rollout["group"].isin(["treated", "control"]).all()
    assert rollout["on_time_rate"].between(0, 1).all()
    assert rollout["n_on_time"].le(rollout["n_customers"]).all()
    assert not rollout.isnull().any().any()


def test_rollout_has_balanced_panel(rollout):
    # every region should appear on every day exactly once
    counts = rollout.groupby("region_id").size()
    assert (counts == 40).all()


def test_treated_regions_improve_after_rollout(rollout):
    treated = rollout[rollout.group == "treated"]
    pre = treated[~treated.post_rollout]["on_time_rate"].mean()
    post = treated[treated.post_rollout]["on_time_rate"].mean()
    assert post > pre


# --- 3. RFM customers ---------------------------------------------------

@pytest.fixture(scope="module")
def rfm_customers():
    return gd.make_rfm_customers(n=1000)


def test_rfm_schema_and_ranges(rfm_customers):
    expected = {"customer_id", "last_order_date", "recency_days", "frequency", "monetary_usd"}
    assert expected.issubset(rfm_customers.columns)
    assert rfm_customers["customer_id"].is_unique
    assert rfm_customers["recency_days"].ge(0).all()
    assert rfm_customers["frequency"].ge(1).all()
    assert rfm_customers["monetary_usd"].gt(0).all()
    assert not rfm_customers.isnull().any().any()


# --- 4. Support tickets --------------------------------------------------

@pytest.fixture(scope="module")
def tickets():
    return gd.make_support_tickets(n=500)


def test_tickets_schema(tickets):
    assert {"ticket_id", "true_topic", "ticket_text"}.issubset(tickets.columns)
    assert tickets["ticket_id"].is_unique
    assert set(tickets["true_topic"]) == set(gd.TOPIC_TEMPLATES.keys())
    assert tickets["ticket_text"].str.len().gt(0).all()


def test_tickets_every_topic_represented(tickets):
    # with n=500 and the smallest topic weight at 0.15, every topic
    # should show up many times - a missing topic would mean a broken
    # sampling step, not just noise.
    counts = tickets["true_topic"].value_counts()
    assert (counts > 10).all()
