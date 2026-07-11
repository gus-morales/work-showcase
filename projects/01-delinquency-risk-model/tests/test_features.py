"""Unit tests for feature engineering: pure functions, hand-computed
expected values on a tiny fixture instead of the full generated dataset."""
import pandas as pd
import pytest

from features import engineer_features, build_design_matrix, CATEGORICAL_COLS


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "loan_amount_usd": [1000.0, 2000.0],
        "monthly_income_usd": [500.0, 0.0],  # second row exercises the clip(lower=1) guard
        "down_payment_ratio": [0.2, 0.0],
        "num_installments": [4, 5],
        "credit_bureau_score": [550, 700],
        "tenure_months_platform": [1, 24],
        "city_tier": ["tier1", "tier3"],
        "employment_type": ["salaried", "informal"],
        "device_type": ["android", "ios"],
        "acquisition_channel": ["organic", "referral"],
        "merchant_category": ["electronics", "travel"],
        "age": [30, 45],
        "num_previous_loans": [2, 10],
        "avg_prior_repayment_delay_days": [1.0, 5.0],
        "num_active_loans_elsewhere": [0, 3],
    })


def test_loan_to_income_ratio(sample_df):
    out = engineer_features(sample_df)
    assert out.loc[0, "loan_to_income_ratio"] == pytest.approx(1000.0 / 500.0)
    # income clipped to a floor of 1 before dividing, so this doesn't blow up to inf
    assert out.loc[1, "loan_to_income_ratio"] == pytest.approx(2000.0 / 1.0)


def test_installment_amount(sample_df):
    out = engineer_features(sample_df)
    expected_0 = 1000.0 * (1 - 0.2) / 4
    assert out.loc[0, "installment_amount_usd"] == pytest.approx(expected_0)


def test_low_bureau_score_flag(sample_df):
    out = engineer_features(sample_df)
    assert out.loc[0, "low_bureau_score"] == 1   # 550 < 580
    assert out.loc[1, "low_bureau_score"] == 0   # 700 >= 580


def test_is_new_customer_flag(sample_df):
    out = engineer_features(sample_df)
    assert out.loc[0, "is_new_customer"] == 1    # tenure 1 month < 3
    assert out.loc[1, "is_new_customer"] == 0    # tenure 24 months


def test_engineer_features_does_not_mutate_input(sample_df):
    original_cols = set(sample_df.columns)
    engineer_features(sample_df)
    assert set(sample_df.columns) == original_cols


def test_build_design_matrix_one_hot_encodes_categoricals(sample_df):
    df = engineer_features(sample_df)
    X, feature_names = build_design_matrix(df)
    for col in CATEGORICAL_COLS:
        assert not any(c == col for c in X.columns)  # raw categorical dropped, only dummies remain
    assert any(c.startswith("city_tier_") for c in X.columns)
    assert X.shape[0] == len(df)
    assert list(X.columns) == feature_names


def test_build_design_matrix_has_no_missing_values(sample_df):
    df = engineer_features(sample_df)
    X, _ = build_design_matrix(df)
    assert not X.isnull().any().any()
