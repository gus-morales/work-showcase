"""Data-generation invariants for the customer/order simulator."""
import pytest

import generate_data as gd


@pytest.fixture(scope="module")
def customers():
    return gd.make_customers(400)


@pytest.fixture(scope="module")
def orders(customers):
    return gd.simulate_transactions(customers)


def test_customers_schema_and_ranges(customers):
    expected_cols = {
        "customer_id", "cohort_month", "acquisition_channel", "city",
        "city_tier", "employment_type", "mean_order_value_usd",
    }
    assert expected_cols.issubset(customers.columns)
    assert customers["customer_id"].is_unique
    assert customers["cohort_month"].between(1, gd.N_MONTHS).all()
    assert customers["city_tier"].isin(["tier1", "tier2", "tier3"]).all()
    assert customers["mean_order_value_usd"].gt(0).all()
    assert not customers.isnull().any().any()


def test_every_customer_has_a_signup_order(customers, orders):
    # simulate_transactions always includes the t=0 signup order, so
    # every customer should appear at least once in orders.
    assert set(customers["customer_id"]) == set(orders["customer_id"])


def test_orders_schema_and_ranges(orders):
    assert orders["order_id"].is_unique
    assert orders["order_value_usd"].gt(0).all()
    assert orders["fee_revenue_usd"].gt(0).all()
    assert orders["months_since_acquisition"].ge(0).all()
    assert orders["order_month_index"].between(1, gd.N_MONTHS).all()
    assert not orders.isnull().any().any()


def test_fee_revenue_matches_take_rate(orders):
    implied_take_rate = (orders["fee_revenue_usd"] / orders["order_value_usd"]).mean()
    assert implied_take_rate == pytest.approx(gd.TAKE_RATE, abs=1e-6)


def test_partner_store_customers_order_more_than_paid_social(customers, orders):
    # This differentiation is the whole point of the channel-quality
    # story in the project; if it collapses the generator is broken.
    merged = orders.merge(customers[["customer_id", "acquisition_channel"]], on="customer_id")
    counts = merged.groupby("acquisition_channel").size() / customers.groupby("acquisition_channel").size()
    assert counts["partner_store"] > counts["paid_social"]
