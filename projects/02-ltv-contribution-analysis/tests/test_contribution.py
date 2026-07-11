"""Unit tests for the log-share GMV decomposition: a pure function, so
these check the arithmetic directly against hand-computed cases rather
than round-tripping through the generated data."""
import numpy as np
import pytest

from contribution import decompose


def test_contributions_sum_to_delta_gmv():
    row_a = {"active_customers": 1000, "orders_per_customer": 2.0, "avg_order_value_usd": 100.0, "gmv_usd": 200_000}
    row_b = {"active_customers": 1500, "orders_per_customer": 1.8, "avg_order_value_usd": 110.0, "gmv_usd": 297_000}
    contributions, delta_gmv = decompose(row_a, row_b)
    assert sum(contributions.values()) == pytest.approx(delta_gmv)


def test_equal_relative_growth_splits_evenly():
    # All three drivers grow by the same multiplicative factor, so each
    # should get exactly a third of the dollar change.
    row_a = {"active_customers": 1000, "orders_per_customer": 2.0, "avg_order_value_usd": 100.0, "gmv_usd": 200_000}
    row_b = {"active_customers": 1200, "orders_per_customer": 2.4, "avg_order_value_usd": 120.0, "gmv_usd": 200_000 * 1.2 ** 3}
    contributions, delta_gmv = decompose(row_a, row_b)
    for v in contributions.values():
        assert v == pytest.approx(delta_gmv / 3)


def test_single_driver_change_gets_full_credit():
    # Only active_customers changes; frequency and order value are flat.
    row_a = {"active_customers": 1000, "orders_per_customer": 2.0, "avg_order_value_usd": 100.0, "gmv_usd": 200_000}
    row_b = {"active_customers": 1500, "orders_per_customer": 2.0, "avg_order_value_usd": 100.0, "gmv_usd": 300_000}
    contributions, delta_gmv = decompose(row_a, row_b)
    assert contributions["customers"] == pytest.approx(delta_gmv)
    assert contributions["frequency"] == pytest.approx(0.0, abs=1e-9)
    assert contributions["avg_order_value"] == pytest.approx(0.0, abs=1e-9)


def test_no_change_falls_back_to_even_split_without_dividing_by_zero():
    row_a = {"active_customers": 1000, "orders_per_customer": 2.0, "avg_order_value_usd": 100.0, "gmv_usd": 200_000}
    row_b = dict(row_a)  # identical period, total_log == 0
    contributions, delta_gmv = decompose(row_a, row_b)
    assert delta_gmv == 0
    assert contributions == {"customers": pytest.approx(0.0), "frequency": pytest.approx(0.0),
                              "avg_order_value": pytest.approx(0.0)}


def test_offsetting_drivers_near_zero_total_log_uses_even_split():
    # Customers up, frequency down by roughly the same log-magnitude, so
    # total_log is near zero even though gmv_usd changed a bit (e.g. from
    # a small avg_order_value move) - exercises the |total_log| < 1e-9 branch.
    row_a = {"active_customers": 1000, "orders_per_customer": 2.0, "avg_order_value_usd": 100.0, "gmv_usd": 200_000}
    row_b = {"active_customers": 1000 * np.e, "orders_per_customer": 2.0 / np.e,
             "avg_order_value_usd": 100.0, "gmv_usd": 205_000}
    contributions, delta_gmv = decompose(row_a, row_b)
    for v in contributions.values():
        assert v == pytest.approx(delta_gmv / 3)
