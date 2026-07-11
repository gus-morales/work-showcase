"""Unit tests for the rate-mix shift bridge decomposition: a pure function
over small hand-built weight/rate series with hand-computed expected
mix and rate effects."""
import pandas as pd
import pytest

from rate_mix_shift import decompose_rate_mix_shift


@pytest.fixture
def two_segment_example():
    # Segment A: weight 0.6 -> 0.5, rate 0.10 -> 0.20 (shrinks, gets much riskier)
    # Segment B: weight 0.4 -> 0.5, rate 0.05 -> 0.05 (grows, rate unchanged)
    weights_a = pd.Series({"A": 0.6, "B": 0.4})
    rates_a = pd.Series({"A": 0.10, "B": 0.05})
    weights_b = pd.Series({"A": 0.5, "B": 0.5})
    rates_b = pd.Series({"A": 0.20, "B": 0.05})
    return weights_a, rates_a, weights_b, rates_b


def test_total_delta_matches_the_direct_weighted_average_change(two_segment_example):
    weights_a, rates_a, weights_b, rates_b = two_segment_example
    result = decompose_rate_mix_shift(weights_a, rates_a, weights_b, rates_b)
    # (0.5*0.20 + 0.5*0.05) - (0.6*0.10 + 0.4*0.05) = 0.125 - 0.08 = 0.045
    assert result["total_delta"] == pytest.approx(0.045)


def test_mix_and_rate_effects_match_hand_computation(two_segment_example):
    weights_a, rates_a, weights_b, rates_b = two_segment_example
    result = decompose_rate_mix_shift(weights_a, rates_a, weights_b, rates_b)
    assert result["mix_total"] == pytest.approx(-0.01)
    assert result["rate_total"] == pytest.approx(0.055)


def test_mix_and_rate_effects_sum_exactly_to_total_delta_no_residual(two_segment_example):
    weights_a, rates_a, weights_b, rates_b = two_segment_example
    result = decompose_rate_mix_shift(weights_a, rates_a, weights_b, rates_b)
    assert result["mix_total"] + result["rate_total"] == pytest.approx(result["total_delta"])


def test_no_mix_shift_at_all_produces_zero_mix_effect():
    # Weights identical between periods -> mix effect must be exactly zero,
    # all of the change should land in the rate effect.
    weights_a = pd.Series({"A": 0.6, "B": 0.4})
    weights_b = pd.Series({"A": 0.6, "B": 0.4})
    rates_a = pd.Series({"A": 0.10, "B": 0.05})
    rates_b = pd.Series({"A": 0.20, "B": 0.15})
    result = decompose_rate_mix_shift(weights_a, rates_a, weights_b, rates_b)
    assert result["mix_total"] == pytest.approx(0.0)
    assert result["rate_total"] == pytest.approx(result["total_delta"])


def test_segment_missing_from_one_period_is_treated_as_zero_weight():
    weights_a = pd.Series({"A": 1.0})
    rates_a = pd.Series({"A": 0.10})
    weights_b = pd.Series({"A": 0.8, "B": 0.2})
    rates_b = pd.Series({"A": 0.10, "B": 0.30})
    result = decompose_rate_mix_shift(weights_a, rates_a, weights_b, rates_b)
    assert set(result["segments"]) == {"A", "B"}
    # A's rate is unchanged, so it contributes zero rate effect, but it still
    # has a mix effect since its weight shrank to make room for B.
    assert result["rate_effect"]["A"] == pytest.approx(0.0)
    assert result["mix_effect"]["A"] == pytest.approx(-0.02)
    # B is a brand-new segment: its full weight and rate both count as change.
    assert result["mix_effect"]["B"] == pytest.approx(0.03)
    assert result["rate_effect"]["B"] == pytest.approx(0.03)
    assert result["total_delta"] == pytest.approx(0.04)
