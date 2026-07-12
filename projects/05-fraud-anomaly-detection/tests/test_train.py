"""Unit tests for the cost-optimal threshold search: a pure function of
predicted probabilities, true labels, and transaction amounts, tested
on small hand-built cases instead of the full trained model."""
import numpy as np
import pytest

from train import find_cost_optimal_threshold, CHARGEBACK_FEE_USD, FALSE_POSITIVE_COST_USD


def test_lower_threshold_chosen_when_missed_fraud_is_very_expensive():
    # One fraud case with a huge dollar amount; missing it should dominate
    # the cost function and pull the optimal threshold down (flag more,
    # even at the cost of some false positives).
    y_true = np.array([1, 0, 0, 0, 0])
    y_prob = np.array([0.2, 0.1, 0.05, 0.15, 0.3])
    amounts = np.array([5000.0, 40.0, 40.0, 40.0, 40.0])
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, amounts)
    # At any threshold below 0.2, the fraud case gets caught; the huge
    # forgone amount should make that region cost-optimal.
    assert best_t < 0.2


def test_higher_threshold_chosen_when_fraud_amount_is_trivial():
    # A fraud case with a tiny amount, scoring just below a large block of
    # genuine transactions; catching it means flagging all of them too,
    # which costs far more than eating the trivial fraud amount plus fee.
    y_true = np.array([1] + [0] * 20)
    y_prob = np.array([0.15] + [0.20] * 20)
    amounts = np.array([1.0] + [40.0] * 20)
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, amounts)
    assert best_t > 0.15


def test_cost_at_threshold_one_equals_summed_fraud_amounts_plus_fees():
    # At threshold 1.0, nothing is ever flagged, so the cost is exactly
    # every fraud case's amount plus the chargeback fee, and zero
    # false-positive cost.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.3, 0.1])
    amounts = np.array([100.0, 200.0, 50.0, 50.0])
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, amounts)
    cost_at_top = costs[-1]  # thresholds[-1] is just under 1.0, nothing flagged
    expected = (100.0 + CHARGEBACK_FEE_USD) + (200.0 + CHARGEBACK_FEE_USD)
    assert cost_at_top == pytest.approx(expected)


def test_cost_at_threshold_near_zero_equals_false_positive_cost_on_genuine_transactions():
    # At a threshold near 0, everything gets flagged, so every genuine
    # transaction becomes a false positive and every fraud case is caught
    # (zero missed-fraud cost).
    y_true = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6])
    amounts = np.array([100.0, 50.0, 50.0, 50.0])
    thresholds, costs, best_t = find_cost_optimal_threshold(y_true, y_prob, amounts)
    cost_at_bottom = costs[0]  # thresholds[0] is just above 0, everything flagged
    expected = 3 * FALSE_POSITIVE_COST_USD
    assert cost_at_bottom == pytest.approx(expected)
