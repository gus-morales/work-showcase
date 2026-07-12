"""Unit tests for the fair lending checks: the disparate impact
calculation and the adverse-action reason-code selection logic, both
pure functions over small hand-built inputs, plus a check that the
protected-class-proxy column is never fed to the model."""
import numpy as np
import pandas as pd
import pytest

from fair_lending import disparate_impact_test, reason_label, top_reasons_from_shap_row
from features import RAW_FEATURE_COLS


def test_demographic_group_is_never_a_model_feature():
    assert "demographic_group" not in RAW_FEATURE_COLS


def test_disparate_impact_ratio_matches_hand_computation():
    # Group A: 40/100 approved (40%). Group B: 50/100 approved (50%, the reference).
    df = pd.DataFrame({
        "demographic_group": ["Group A"] * 100 + ["Group B"] * 100,
        "approved": [True] * 40 + [False] * 60 + [True] * 50 + [False] * 50,
    })
    rates, reference_group, results = disparate_impact_test(df)
    assert reference_group == "Group B"
    assert rates.loc["Group A", "approval_rate"] == pytest.approx(0.40)
    assert rates.loc["Group B", "approval_rate"] == pytest.approx(0.50)
    assert results["Group A"]["disparate_impact_ratio"] == pytest.approx(0.80)


def test_four_fifths_rule_boundary_cases():
    # Exactly 0.80 ratio should pass (>=, not strict >).
    df_boundary = pd.DataFrame({
        "demographic_group": ["A"] * 100 + ["B"] * 100,
        "approved": [True] * 40 + [False] * 60 + [True] * 50 + [False] * 50,
    })
    _, _, results = disparate_impact_test(df_boundary)
    assert results["A"]["passes_four_fifths_rule"] is True

    # A clearly larger gap should fail.
    df_fail = pd.DataFrame({
        "demographic_group": ["A"] * 100 + ["B"] * 100,
        "approved": [True] * 30 + [False] * 70 + [True] * 50 + [False] * 50,
    })
    _, _, results_fail = disparate_impact_test(df_fail)
    assert results_fail["A"]["passes_four_fifths_rule"] is False


def test_identical_approval_rates_give_ratio_of_one_and_no_significant_gap():
    df = pd.DataFrame({
        "demographic_group": ["A"] * 200 + ["B"] * 200,
        "approved": ([True] * 80 + [False] * 120) * 2,
    })
    _, reference_group, results = disparate_impact_test(df)
    # With identical rates, whichever group isn't picked as the reference
    # (idxmax's tie-break) should show a ratio of exactly 1.0 against it.
    non_reference = "A" if reference_group == "B" else "B"
    assert results[non_reference]["disparate_impact_ratio"] == pytest.approx(1.0)
    assert results[non_reference]["statistically_significant_gap"] is False


def test_reason_label_maps_known_prefixes():
    assert reason_label("credit_bureau_score") == "Credit score"
    assert reason_label("credit_bureau_score_na") == "Insufficient credit history on file"
    assert reason_label("low_bureau_score") == "Credit score"
    assert reason_label("avg_prior_repayment_delay_days") == "Delinquent past credit obligations"


def test_reason_label_excludes_geography_channel_and_merchant():
    # These carry real SHAP signal in the model but are deliberately
    # excluded from the reason-code universe.
    assert reason_label("city_tier_tier1") is None
    assert reason_label("device_type_android") is None
    assert reason_label("acquisition_channel_organic") is None
    assert reason_label("merchant_category_electronics") is None


def test_reason_label_longest_prefix_wins_for_bureau_score_na():
    # credit_bureau_score_na literally starts with "credit_bureau_score";
    # the missing-indicator label must win, not the generic score label.
    assert reason_label("credit_bureau_score_na") == "Insufficient credit history on file"
    assert reason_label("credit_bureau_score_na") != reason_label("credit_bureau_score")


def test_top_reasons_from_shap_row_ranks_by_value_and_keeps_only_positive():
    row = np.array([0.5, -0.2, 0.1, 0.8, 0.0])
    labels = ["Credit score", "Income insufficient for amount of credit requested",
              "Down payment amount", "Delinquent past credit obligations", "Term of credit requested"]
    result = top_reasons_from_shap_row(row, labels, top_n=3)
    # Sorted descending by value: 0.8 (Delinquent...), 0.5 (Credit score), 0.1 (Down payment).
    # Negative and zero values are excluded entirely.
    assert result == ["Delinquent past credit obligations", "Credit score", "Down payment amount"]


def test_top_reasons_from_shap_row_deduplicates_same_label():
    row = np.array([0.5, 0.4, 0.3])
    labels = ["Credit score", "Credit score", "Down payment amount"]
    result = top_reasons_from_shap_row(row, labels, top_n=3)
    assert result == ["Credit score", "Down payment amount"]


def test_top_reasons_from_shap_row_returns_fewer_than_top_n_if_not_enough_positive_reasons():
    row = np.array([0.3, -0.1, -0.5])
    labels = ["Credit score", "Down payment amount", "Term of credit requested"]
    result = top_reasons_from_shap_row(row, labels, top_n=3)
    assert result == ["Credit score"]


def test_top_reasons_from_shap_row_returns_empty_list_when_no_positive_contributions():
    row = np.array([-0.1, -0.2, 0.0])
    labels = ["Credit score", "Down payment amount", "Term of credit requested"]
    assert top_reasons_from_shap_row(row, labels, top_n=3) == []
