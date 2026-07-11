"""Unit tests for the data contracts, against small hand-built
customers/orders tables covering both a clean pass and one violation
per rule at a time, so each check is exercised in isolation."""
import pandas as pd
import pytest

from contracts import (
    DataContractError, validate_customers, validate_orders,
    validate_referential_integrity, run_all_contracts,
)


@pytest.fixture
def clean_customers():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "cohort_month": [1, 2, 24],
        "acquisition_channel": ["organic", "referral", "paid_social"],
        "city_tier": ["tier1", "tier2", "tier3"],
        "employment_type": ["salaried", "gig_economy", "informal"],
        "mean_order_value_usd": [500.0, 420.0, 310.0],
    })


@pytest.fixture
def clean_orders():
    return pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": [1, 2, 3],
        "order_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
        "order_value_usd": [500.0, 420.0, 310.0],
        "months_since_acquisition": [0, 0, 0],
        "order_month_index": [1, 2, 24],
        "fee_revenue_usd": [32.5, 27.3, 20.15],
    })


def test_clean_data_passes_all_contracts(clean_customers, clean_orders):
    assert validate_customers(clean_customers) == []
    assert validate_orders(clean_orders) == []
    assert validate_referential_integrity(clean_customers, clean_orders) == []
    run_all_contracts(clean_customers, clean_orders)  # should not raise


def test_missing_column_is_reported_once(clean_customers):
    broken = clean_customers.drop(columns=["acquisition_channel"])
    violations = validate_customers(broken)
    assert any("missing required column 'acquisition_channel'" in v for v in violations)


def test_duplicate_customer_id_is_flagged(clean_customers):
    broken = clean_customers.copy()
    broken.loc[1, "customer_id"] = broken.loc[0, "customer_id"]
    violations = validate_customers(broken)
    assert any("duplicates" in v for v in violations)


def test_unexpected_category_value_is_flagged(clean_customers):
    broken = clean_customers.copy()
    broken.loc[0, "acquisition_channel"] = "influencer"
    violations = validate_customers(broken)
    assert any("acquisition_channel" in v and "influencer" in v for v in violations)


def test_cohort_month_out_of_range_is_flagged(clean_customers):
    broken = clean_customers.copy()
    broken.loc[0, "cohort_month"] = 99
    violations = validate_customers(broken)
    assert any("cohort_month" in v for v in violations)


def test_negative_order_value_is_flagged(clean_orders):
    broken = clean_orders.copy()
    broken.loc[0, "order_value_usd"] = -10.0
    violations = validate_orders(broken)
    assert any("order_value_usd" in v and "non-positive" in v for v in violations)


def test_fee_exceeding_order_value_is_flagged(clean_orders):
    broken = clean_orders.copy()
    broken.loc[0, "fee_revenue_usd"] = broken.loc[0, "order_value_usd"] + 5
    violations = validate_orders(broken)
    assert any("fee_revenue_usd >= order_value_usd" in v for v in violations)


def test_orphan_customer_id_is_flagged(clean_customers, clean_orders):
    broken = clean_orders.copy()
    broken.loc[0, "customer_id"] = 999
    violations = validate_referential_integrity(clean_customers, broken)
    assert any("999" in v for v in violations)


def test_run_all_contracts_raises_with_all_violations_listed(clean_customers, clean_orders):
    broken_orders = clean_orders.copy()
    broken_orders.loc[0, "order_value_usd"] = -10.0
    broken_orders.loc[1, "customer_id"] = 999
    with pytest.raises(DataContractError) as exc_info:
        run_all_contracts(clean_customers, broken_orders)
    message = str(exc_info.value)
    assert "non-positive" in message
    assert "999" in message
