"""Unit tests for the cost-optimal threshold search, on a tiny hand-built
set of (risk_score, true_bad) pairs where the minimum-cost threshold can
be worked out by hand."""
import numpy as np
import pytest

from guardrail_threshold import find_cost_optimal_threshold, COST_BAD_AUTO_SEND, COST_HUMAN_REVIEW


@pytest.fixture
def sample_scores():
    risk_score = np.array([0.10, 0.30, 0.60, 0.90])
    true_bad = np.array([0, 0, 1, 1])
    return risk_score, true_bad


def test_cost_matches_hand_computation_at_each_threshold(sample_scores):
    risk_score, true_bad = sample_scores
    thresholds = np.array([0.05, 0.35, 0.65, 0.95])
    _, costs, best_t, best_cost = find_cost_optimal_threshold(risk_score, true_bad, thresholds)

    # t=0.05: all 4 routed -> 4 * review cost
    assert costs[0] == pytest.approx(4 * COST_HUMAN_REVIEW)
    # t=0.35: items 3,4 routed (2 * review cost); items 1,2 auto-sent, both good -> no bad-send cost
    assert costs[1] == pytest.approx(2 * COST_HUMAN_REVIEW)
    # t=0.65: item 4 routed (1 * review cost); item 3 (bad) auto-sent -> + bad-send cost
    assert costs[2] == pytest.approx(1 * COST_HUMAN_REVIEW + COST_BAD_AUTO_SEND)
    # t=0.95: none routed; items 3 and 4 (both bad) auto-sent -> 2 * bad-send cost
    assert costs[3] == pytest.approx(2 * COST_BAD_AUTO_SEND)


def test_best_threshold_is_the_minimum_cost_one(sample_scores):
    risk_score, true_bad = sample_scores
    thresholds = np.array([0.05, 0.35, 0.65, 0.95])
    _, _, best_t, best_cost = find_cost_optimal_threshold(risk_score, true_bad, thresholds)
    assert best_t == pytest.approx(0.35)
    assert best_cost == pytest.approx(2 * COST_HUMAN_REVIEW)


def test_default_threshold_grid_has_99_points(sample_scores):
    risk_score, true_bad = sample_scores
    thresholds, costs, _, _ = find_cost_optimal_threshold(risk_score, true_bad)
    assert len(thresholds) == 99
    assert len(costs) == 99
