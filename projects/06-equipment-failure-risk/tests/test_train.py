"""Unit tests for the cost-optimal threshold search: a pure function of
predicted probabilities, true labels, and downtime costs, tested on
small hand-built cases instead of the full trained model."""
import numpy as np
import pytest

from train import find_cost_optimal_threshold, UNPLANNED_FAILURE_REPAIR_PREMIUM_USD, SCHEDULED_INSPECTION_COST_USD


def test_lower_threshold_chosen_when_missed_failure_is_very_expensive():
    # One failure case with a huge downtime cost; missing it should
    # dominate the cost function and pull the optimal threshold down
    # (flag more, even at the cost of some false positives).
    y_true = np.array([1, 0, 0, 0, 0])
    y_prob = np.array([0.2, 0.1, 0.05, 0.15, 0.3])
    downtime_costs = np.array([150_000.0, 1_000.0, 1_000.0, 1_000.0, 1_000.0])
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, downtime_costs)
    # At any threshold below 0.2, the failure case gets caught; the huge
    # forgone downtime cost should make that region cost-optimal.
    assert best_t < 0.2


def test_higher_threshold_chosen_when_failure_cost_is_trivial():
    # A failure case with a tiny downtime cost, scoring just below a
    # large block of healthy truck-days; catching it means flagging all
    # of them too, which costs far more than eating the trivial
    # downtime cost plus premium.
    y_true = np.array([1] + [0] * 20)
    y_prob = np.array([0.15] + [0.20] * 20)
    downtime_costs = np.array([100.0] + [1_000.0] * 20)
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, downtime_costs)
    assert best_t > 0.15


def test_cost_at_threshold_one_equals_summed_downtime_costs_plus_premium():
    # At threshold 1.0, nothing is ever flagged, so the cost is exactly
    # every failure case's downtime cost plus the repair premium, and
    # zero false-positive cost.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.3, 0.1])
    downtime_costs = np.array([10_000.0, 20_000.0, 5_000.0, 5_000.0])
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, downtime_costs)
    cost_at_top = costs[-1]  # thresholds[-1] is just under 1.0, nothing flagged
    expected = (10_000.0 + UNPLANNED_FAILURE_REPAIR_PREMIUM_USD) + (20_000.0 + UNPLANNED_FAILURE_REPAIR_PREMIUM_USD)
    assert cost_at_top == pytest.approx(expected)


def test_cost_at_threshold_near_zero_equals_false_positive_cost_on_healthy_truck_days():
    # At a threshold near 0, everything gets flagged, so every healthy
    # truck-day becomes a false positive and every failure is caught
    # (zero missed-failure cost).
    y_true = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6])
    downtime_costs = np.array([10_000.0, 5_000.0, 5_000.0, 5_000.0])
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, downtime_costs)
    cost_at_bottom = costs[0]  # thresholds[0] is just above 0, everything flagged
    expected = 3 * SCHEDULED_INSPECTION_COST_USD
    assert cost_at_bottom == pytest.approx(expected)
