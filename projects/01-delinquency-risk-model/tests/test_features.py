"""Unit tests for feature engineering: pure functions and the
feature-engine pipeline, on a tiny fixture with hand-computed expected
values instead of the full generated dataset."""
import numpy as np
import pandas as pd
import pytest

from features import engineer_features, build_feature_pipeline, CATEGORICAL_COLS, RAW_FEATURE_COLS


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


@pytest.fixture
def sample_df_with_missing_bureau_score(sample_df):
    df = sample_df.copy()
    df["credit_bureau_score"] = [np.nan, 700.0]
    return df


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


def test_low_bureau_score_flag_is_zero_when_score_is_missing(sample_df_with_missing_bureau_score):
    out = engineer_features(sample_df_with_missing_bureau_score)
    # NaN < 580 is False: "unknown" is not the same thing as "known-low",
    # and gets its own signal from the pipeline's missing indicator instead.
    assert out.loc[0, "low_bureau_score"] == 0


def test_is_new_customer_flag(sample_df):
    out = engineer_features(sample_df)
    assert out.loc[0, "is_new_customer"] == 1    # tenure 1 month < 3
    assert out.loc[1, "is_new_customer"] == 0    # tenure 24 months


def test_engineer_features_does_not_mutate_input(sample_df):
    original_cols = set(sample_df.columns)
    engineer_features(sample_df)
    assert set(sample_df.columns) == original_cols


def test_pipeline_one_hot_encodes_categoricals(sample_df):
    df = engineer_features(sample_df)
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df[RAW_FEATURE_COLS])
    for col in CATEGORICAL_COLS:
        assert not any(c == col for c in X.columns)  # raw categorical dropped, only dummies remain
    assert any(c.startswith("city_tier_") for c in X.columns)
    assert X.shape[0] == len(df)


def test_pipeline_has_no_missing_values_on_complete_data(sample_df):
    df = engineer_features(sample_df)
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df[RAW_FEATURE_COLS])
    assert not X.isnull().any().any()


def test_pipeline_imputes_missing_bureau_score_and_adds_indicator(sample_df_with_missing_bureau_score):
    df = engineer_features(sample_df_with_missing_bureau_score)
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df[RAW_FEATURE_COLS])
    assert "credit_bureau_score_na" in X.columns
    assert X.loc[0, "credit_bureau_score_na"] == 1
    assert X.loc[1, "credit_bureau_score_na"] == 0
    # Imputed with the (training-split) median: with one missing and one
    # observed value (700), the median of the observed value is 700.
    assert X.loc[0, "credit_bureau_score"] == pytest.approx(700.0)
    assert not X["credit_bureau_score"].isnull().any()


def test_pipeline_fit_on_train_then_transform_reuses_train_statistics():
    # A missing value in the "test" split should be imputed with the
    # *training* split's median, not recomputed from the test split alone.
    train = pd.DataFrame({
        "credit_bureau_score": [600.0, 620.0, 640.0],
        "loan_amount_usd": [1000.0, 1200.0, 1100.0],
        "monthly_income_usd": [500.0, 600.0, 550.0],
        "city_tier": ["tier1", "tier2", "tier1"],
    })
    test = pd.DataFrame({
        "credit_bureau_score": [np.nan],
        "loan_amount_usd": [1050.0],
        "monthly_income_usd": [520.0],
        "city_tier": ["tier1"],
    })
    from feature_engine.imputation import MeanMedianImputer
    imputer = MeanMedianImputer(imputation_method="median", variables=["credit_bureau_score"])
    imputer.fit(train)
    out = imputer.transform(test)
    assert out.loc[0, "credit_bureau_score"] == pytest.approx(620.0)  # median of the train split
