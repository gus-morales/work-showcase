"""Unit tests for the power-analysis helper and the two-proportion A/B
result computation, both pure functions independent of plotting."""
import pandas as pd
import pytest

from ab_test import cohens_h, required_n_per_arm, compute_ab_result


def test_cohens_h_is_zero_at_equal_proportions():
    assert cohens_h(0.5, 0.5) == pytest.approx(0.0)
    assert cohens_h(0.34, 0.34) == pytest.approx(0.0)


def test_cohens_h_is_antisymmetric():
    h = cohens_h(0.3, 0.5)
    assert cohens_h(0.5, 0.3) == pytest.approx(-h)


def test_required_n_decreases_as_mde_grows():
    n_small_effect = required_n_per_arm(mde=0.02)
    n_large_effect = required_n_per_arm(mde=0.08)
    assert n_large_effect < n_small_effect


@pytest.fixture
def sample_ab_df():
    # v1_baseline: 6/10 acceptable (0.6). v2_revised: 8/10 acceptable (0.8).
    return pd.DataFrame({
        "arm": ["v1_baseline"] * 10 + ["v2_revised"] * 10,
        "judge_acceptable": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    })


def test_compute_ab_result_matches_hand_computed_rates(sample_ab_df):
    result = compute_ab_result(sample_ab_df)
    assert result["p_v1"] == pytest.approx(0.6)
    assert result["p_v2"] == pytest.approx(0.8)
    assert result["lift_abs"] == pytest.approx(0.2)
    assert result["n_v1"] == 10
    assert result["n_v2"] == 10


def test_compute_ab_result_ci_diff_brackets_the_point_estimate(sample_ab_df):
    result = compute_ab_result(sample_ab_df)
    lo, hi = result["ci_diff"]
    assert lo < result["lift_abs"] < hi
