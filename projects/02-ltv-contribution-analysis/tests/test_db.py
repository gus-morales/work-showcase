"""Runs every .sql file in sql/ against a small generated dataset written
to a temp dir (not the project's real data/ folder), checking each query
executes cleanly and returns the columns downstream code expects."""

import pytest

import generate_data as gd
from db import get_connection, run_sql_file


@pytest.fixture(scope="module")
def temp_data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    customers = gd.make_customers(300)
    orders = gd.simulate_transactions(customers)
    customers.to_csv(d / "customers.csv", index=False)
    orders.to_csv(d / "orders.csv", index=False)
    return d


@pytest.fixture(scope="module")
def con(temp_data_dir):
    return get_connection(temp_data_dir)


def test_cohort_revenue_query(con):
    df = run_sql_file(con, "01_cohort_revenue.sql")
    assert not df.empty
    assert {"cohort_month", "months_since_acquisition", "retention_rate",
            "revenue_per_acquired_customer"}.issubset(df.columns)
    assert df["retention_rate"].between(0, 1.0001).all()


def test_monthly_kpis_query(con):
    df = run_sql_file(con, "02_monthly_kpis.sql")
    assert not df.empty
    assert {"month_index", "active_customers", "orders", "gmv_usd",
            "avg_order_value_usd", "orders_per_customer"}.issubset(df.columns)
    assert df["active_customers"].gt(0).all()


def test_channel_quality_query(con):
    df = run_sql_file(con, "03_channel_quality.sql")
    assert not df.empty
    assert {"acquisition_channel", "avg_orders_per_customer",
            "avg_revenue_per_customer_usd"}.issubset(df.columns)


def test_channel_mix_shift_query(con):
    df = run_sql_file(con, "04_channel_mix_shift.sql")
    assert not df.empty
    # shares within a cohort_month should sum to ~1
    sums = df.groupby("cohort_month")["channel_share"].sum()
    assert sums.round(2).between(0.99, 1.01).all()
