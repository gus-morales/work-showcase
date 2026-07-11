"""Unit tests for the judge-vs-human validation logic: a pure function
over a small hand-built label set with known, hand-computed values."""
import pandas as pd
import pytest

from eval_framework import compute_judge_validation


@pytest.fixture
def sample_df():
    # category "a": no bias (diffs 0, 0)
    # category "b": mild bias (diffs 1, 0)
    # category "c": large bias (diffs 2, 1)
    return pd.DataFrame({
        "ticket_id": [1, 2, 3, 4, 5, 6],
        "category": ["a", "a", "b", "b", "c", "c"],
        "human_label": [3, 3, 3, 3, 2, 2],
        "judge_label": [3, 3, 4, 3, 4, 3],
    })


def test_exact_agreement_rate(sample_df):
    result = compute_judge_validation(sample_df)
    # matches: rows 1, 2, 4 -> 3 of 6
    assert result["exact_agreement_rate"] == pytest.approx(3 / 6)


def test_adjacent_agreement_rate(sample_df):
    result = compute_judge_validation(sample_df)
    # only row 5 has |diff| > 1 (human=2, judge=4)
    assert result["adjacent_agreement_rate"] == pytest.approx(5 / 6)


def test_bias_overall_matches_hand_computed_mean(sample_df):
    result = compute_judge_validation(sample_df)
    # diffs: 0, 0, 1, 0, 2, 1 -> mean = 4/6
    assert result["bias_overall"] == pytest.approx(4 / 6)


def test_bias_by_category_ranks_c_highest_and_a_lowest(sample_df):
    result = compute_judge_validation(sample_df)
    bias = result["bias_by_category"]
    assert bias.index[0] == "c"
    assert bias.index[-1] == "a"
    assert bias["a"] == pytest.approx(0.0)
    assert bias["b"] == pytest.approx(0.5)
    assert bias["c"] == pytest.approx(1.5)


def test_kappa_and_correlation_are_in_valid_range(sample_df):
    result = compute_judge_validation(sample_df)
    assert -1.0 <= result["kappa"] <= 1.0
    assert -1.0 <= result["correlation"] <= 1.0


def test_bias_p_value_is_a_valid_probability(sample_df):
    result = compute_judge_validation(sample_df)
    assert 0.0 <= result["bias_p_value"] <= 1.0
