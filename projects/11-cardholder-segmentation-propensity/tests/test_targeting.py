"""Unit tests for the budget-constrained expected-responders
computation: a pure function over a dataframe with a rank column and a
propensity_score column, checked against hand-computed totals."""
import pandas as pd
import pytest

from targeting import expected_responders


def test_top_budget_by_score_sums_the_highest_scores():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "propensity_score": [0.9, 0.1, 0.7, 0.3, 0.5],
        "monetary_90d": [10, 500, 20, 400, 30],
    })
    # Top 2 by propensity_score are 0.9 and 0.7 -> sum 1.6
    result = expected_responders(df, "propensity_score", budget_n=2)
    assert result == pytest.approx(1.6)


def test_top_budget_by_a_different_rank_column_uses_that_ranking_not_score():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "propensity_score": [0.9, 0.1, 0.7, 0.3, 0.5],
        "monetary_90d": [10, 500, 20, 400, 30],
    })
    # Top 2 by monetary_90d are customers 2 and 4 (scores 0.1 and 0.3) -> sum 0.4
    result = expected_responders(df, "monetary_90d", budget_n=2)
    assert result == pytest.approx(0.4)


def test_full_budget_equals_the_sum_of_all_scores():
    df = pd.DataFrame({
        "propensity_score": [0.9, 0.1, 0.7, 0.3, 0.5],
        "monetary_90d": [10, 500, 20, 400, 30],
    })
    result = expected_responders(df, "propensity_score", budget_n=len(df))
    assert result == pytest.approx(df["propensity_score"].sum())


def test_model_ranked_targeting_never_captures_less_than_random_rank_on_same_budget():
    # Ranking by propensity_score itself is definitionally the
    # score-maximizing choice for a fixed budget: no other ranking of
    # the same column set can beat it.
    df = pd.DataFrame({
        "propensity_score": [0.9, 0.1, 0.7, 0.3, 0.5, 0.2],
        "monetary_90d": [10, 500, 20, 400, 30, 1000],
    })
    by_score = expected_responders(df, "propensity_score", budget_n=3)
    by_spend = expected_responders(df, "monetary_90d", budget_n=3)
    assert by_score >= by_spend
