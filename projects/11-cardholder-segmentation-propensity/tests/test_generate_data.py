"""Data-generation invariants: schema, ranges, and the directional
relationships the response label is supposed to encode. Not testing
exact values (the generator is stochastic by design), just the
contract the rest of the pipeline relies on, plus that the "win-back
sweet spot" and the historical-value proxy actually raise response
propensity, since that's exactly what propensity_model.py and
interpret.py are supposed to recover."""
import numpy as np
import pandas as pd
import pytest

import generate_data as gd


@pytest.fixture(scope="module")
def customers():
    return gd.make_customers(4000)


@pytest.fixture(scope="module")
def with_behavior(customers):
    return gd.add_behavioral_features(customers)


@pytest.fixture(scope="module")
def full(with_behavior):
    return gd.assign_offer_and_response(with_behavior)


def test_customers_schema_and_ranges(customers):
    assert customers["customer_id"].is_unique
    assert customers["tenure_days"].between(30, 1500).all()
    assert (customers["lifetime_orders"] >= 1).all()
    assert not customers.isnull().any().any()


def test_behavioral_features_schema_and_ranges(with_behavior):
    assert with_behavior["recency_days"].between(0, 180).all()
    assert (with_behavior["frequency_90d"] >= 0).all()
    assert (with_behavior["monetary_90d"] >= 0).all()
    assert with_behavior["category_diversity"].between(1, 6).all()
    # category diversity can never exceed how many orders were placed
    assert (with_behavior["category_diversity"] <= np.maximum(with_behavior["frequency_90d"], 1)).all()
    assert with_behavior["decline_rate"].between(0.01, 0.60).all()
    assert with_behavior["primary_channel"].isin(gd.CHANNELS).all()
    assert not with_behavior.isnull().any().any()


def test_offer_flag_is_binary(full):
    assert full["past_offer_sent"].isin([0, 1]).all()


def test_responded_is_null_iff_never_offered(full):
    # "never offered" and "offered but didn't respond" are different
    # states: responded should be null exactly where past_offer_sent
    # is 0, and a real 0/1 value everywhere it's 1.
    never_offered = full["past_offer_sent"] == 0
    assert full.loc[never_offered, "responded"].isna().all()
    assert full.loc[~never_offered, "responded"].isin([0.0, 1.0]).all()
    assert not full.loc[~never_offered, "responded"].isna().any()


def test_offer_rate_and_response_rate_are_plausible(full):
    offer_rate = full["past_offer_sent"].mean()
    response_rate = full.loc[full["past_offer_sent"] == 1, "responded"].mean()
    # Loose sanity bands: catch a broken generator (inverted sign,
    # runaway intercept) without being flaky about the exact rate.
    assert 0.25 < offer_rate < 0.65
    assert 0.15 < response_rate < 0.50


def test_lapsed_band_responds_better_than_active_or_dormant(full):
    offered = full[full["past_offer_sent"] == 1]
    lapsed = offered[(offered["recency_days"] >= 30) & (offered["recency_days"] <= 120)]
    active = offered[offered["recency_days"] < 30]
    dormant = offered[offered["recency_days"] > 120]
    assert lapsed["responded"].mean() > active["responded"].mean()
    assert lapsed["responded"].mean() > dormant["responded"].mean()


def test_higher_lifetime_orders_raises_response_rate(full):
    # lifetime_orders is a proxy for the latent baseline_value trait
    # response is actually generated from; higher orders should mean
    # higher response propensity even though baseline_value itself is
    # never observed directly.
    offered = full[full["past_offer_sent"] == 1]
    median = offered["lifetime_orders"].median()
    high_rate = offered.loc[offered["lifetime_orders"] > median, "responded"].mean()
    low_rate = offered.loc[offered["lifetime_orders"] <= median, "responded"].mean()
    assert high_rate > low_rate


def test_decline_rate_is_a_weaker_signal_than_lifetime_orders(full):
    # decline_rate has zero true weight in the response formula (a
    # decoy feature); the gap in response rate across its tercile split
    # should be much smaller than the gap across lifetime_orders'.
    offered = full[full["past_offer_sent"] == 1].copy()
    offered["decline_tercile"] = pd.qcut(offered["decline_rate"], 3, labels=["low", "mid", "high"])
    decline_gap = abs(
        offered.loc[offered["decline_tercile"] == "high", "responded"].mean()
        - offered.loc[offered["decline_tercile"] == "low", "responded"].mean()
    )
    median = offered["lifetime_orders"].median()
    value_gap = abs(
        offered.loc[offered["lifetime_orders"] > median, "responded"].mean()
        - offered.loc[offered["lifetime_orders"] <= median, "responded"].mean()
    )
    assert decline_gap < value_gap
