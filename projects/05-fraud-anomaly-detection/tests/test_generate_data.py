"""Data-generation invariants: schema, ranges, and no-null checks on a
small synthetic sample. Not testing exact values (the generator is
stochastic by design), just the contract the rest of the pipeline
relies on, plus the directional relationships the fraud label is
supposed to encode (new device, mismatches, and velocity should all
raise the fraud rate, since those are exactly the signals the model in
train.py is supposed to recover in interpret.py)."""
import pytest

import generate_data as gd


@pytest.fixture(scope="module")
def customers():
    return gd.make_customers(2000)


@pytest.fixture(scope="module")
def transactions(customers):
    tx = gd.make_transactions(customers)
    tx = gd.add_velocity_features(tx)
    return tx


@pytest.fixture(scope="module")
def full(transactions):
    return gd.assign_fraud_label(transactions)


def test_customers_schema_and_ranges(customers):
    expected_cols = {"customer_id", "account_age_at_start_days", "home_device", "spend_scale", "n_transactions"}
    assert expected_cols.issubset(customers.columns)
    assert customers["customer_id"].is_unique
    assert customers["account_age_at_start_days"].between(0, 1500).all()
    assert customers["home_device"].isin(gd.DEVICE_TYPES).all()
    assert customers["n_transactions"].between(1, 60).all()
    assert not customers.isnull().any().any()


def test_transactions_schema_and_ranges(transactions):
    assert transactions["transaction_id"].is_unique
    assert transactions["amount_usd"].between(3, 6000).all()
    assert transactions["checkout_seconds"].between(2, 400).all()
    assert transactions["merchant_category"].isin(gd.MERCHANT_CATS).all()
    assert transactions["device_type"].isin(gd.DEVICE_TYPES).all()
    assert transactions["is_new_device"].isin([0, 1]).all()
    assert transactions["billing_shipping_mismatch"].isin([0, 1]).all()
    assert transactions["ip_billing_country_mismatch"].isin([0, 1]).all()
    assert not transactions.isnull().any().any()


def test_transactions_reference_valid_customers(transactions, customers):
    assert set(transactions["customer_id"]).issubset(set(customers["customer_id"]))


def test_velocity_counts_are_nonnegative_and_nested(transactions):
    assert (transactions["transactions_last_1h"] >= 0).all()
    assert (transactions["transactions_last_24h"] >= 0).all()
    # The 24h window strictly contains the 1h window, so its count can
    # never be smaller.
    assert (transactions["transactions_last_24h"] >= transactions["transactions_last_1h"]).all()


def test_is_new_device_rate_is_plausible(transactions):
    # ~8% of transactions use a device other than the customer's home
    # device, by construction; loose band, not a tight statistical check.
    rate = transactions["is_new_device"].mean()
    assert 0.03 < rate < 0.15


def test_fraud_flag_is_binary(full):
    assert full["is_fraud"].isin([0, 1]).all()


def test_fraud_rate_is_plausible(full):
    rate = full["is_fraud"].mean()
    # Loose sanity band: catches a broken generator (inverted sign,
    # runaway intercept) without being flaky about the exact rate.
    assert 0.003 < rate < 0.05


def test_new_device_raises_fraud_rate(full):
    new_device_rate = full.loc[full.is_new_device == 1, "is_fraud"].mean()
    recognized_rate = full.loc[full.is_new_device == 0, "is_fraud"].mean()
    assert new_device_rate > recognized_rate


def test_billing_shipping_mismatch_raises_fraud_rate(full):
    mismatch_rate = full.loc[full.billing_shipping_mismatch == 1, "is_fraud"].mean()
    match_rate = full.loc[full.billing_shipping_mismatch == 0, "is_fraud"].mean()
    assert mismatch_rate > match_rate


def test_ip_billing_country_mismatch_raises_fraud_rate(full):
    mismatch_rate = full.loc[full.ip_billing_country_mismatch == 1, "is_fraud"].mean()
    match_rate = full.loc[full.ip_billing_country_mismatch == 0, "is_fraud"].mean()
    assert mismatch_rate > match_rate


def test_recent_velocity_raises_fraud_rate(full):
    busy_rate = full.loc[full.transactions_last_1h >= 1, "is_fraud"].mean()
    quiet_rate = full.loc[full.transactions_last_1h == 0, "is_fraud"].mean()
    assert busy_rate > quiet_rate


def test_older_accounts_have_lower_fraud_rate(full):
    old_rate = full.loc[full.account_age_days_at_tx > 365, "is_fraud"].mean()
    new_rate = full.loc[full.account_age_days_at_tx <= 30, "is_fraud"].mean()
    assert new_rate > old_rate
