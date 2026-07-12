"""Unit tests for the feature pipeline and the temporal split, on tiny
hand-built fixtures instead of the full generated dataset."""
import pandas as pd
import pytest

from features import build_feature_pipeline, temporal_split, CATEGORICAL_COLS, RAW_FEATURE_COLS


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "account_age_days_at_tx": [10, 400],
        "amount_usd": [50.0, 8000.0],  # second row exercises the winsorizer's upper cap
        "checkout_seconds": [5.0, 60.0],
        "transactions_last_1h": [3, 0],
        "transactions_last_24h": [5, 0],
        "is_new_device": [1, 0],
        "billing_shipping_mismatch": [1, 0],
        "ip_billing_country_mismatch": [0, 0],
        "merchant_category": ["electronics", "groceries"],
        "device_type": ["android", "ios"],
    })


def test_pipeline_one_hot_encodes_categoricals(sample_df):
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(sample_df[RAW_FEATURE_COLS])
    for col in CATEGORICAL_COLS:
        assert not any(c == col for c in X.columns)
    assert any(c.startswith("merchant_category_") for c in X.columns)
    assert any(c.startswith("device_type_") for c in X.columns)
    assert X.shape[0] == len(sample_df)


def test_pipeline_winsorizes_extreme_amount():
    # A training split with a tight, low-variance range; a wildly larger
    # test-split value should get capped rather than passed through raw.
    train = pd.DataFrame({
        "account_age_days_at_tx": [100, 110, 105, 95, 102, 98],
        "amount_usd": [40.0, 42.0, 45.0, 38.0, 41.0, 43.0],
        "checkout_seconds": [30.0, 32.0, 28.0, 31.0, 29.0, 33.0],
        "transactions_last_1h": [0] * 6,
        "transactions_last_24h": [0] * 6,
        "is_new_device": [0] * 6,
        "billing_shipping_mismatch": [0] * 6,
        "ip_billing_country_mismatch": [0] * 6,
        "merchant_category": ["electronics"] * 6,
        "device_type": ["android"] * 6,
    })
    test = train.iloc[:1].copy()
    test["amount_usd"] = 9000.0

    pipeline = build_feature_pipeline()
    pipeline.fit(train[RAW_FEATURE_COLS])
    X_test = pipeline.transform(test[RAW_FEATURE_COLS])
    assert X_test.loc[0, "amount_usd"] < 9000.0


def test_pipeline_has_no_missing_values(sample_df):
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(sample_df[RAW_FEATURE_COLS])
    assert not X.isnull().any().any()


def test_temporal_split_partitions_are_disjoint_and_complete():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime("2025-01-01") + pd.to_timedelta(range(180), unit="D"),
        "value": range(180),
    })
    train, val, test = temporal_split(df, train_end_day=100, val_end_day=140)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train["value"]) & set(val["value"]) == set()
    assert set(val["value"]) & set(test["value"]) == set()


def test_temporal_split_train_precedes_val_precedes_test():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime("2025-01-01") + pd.to_timedelta(range(180), unit="D"),
        "value": range(180),
    })
    train, val, test = temporal_split(df, train_end_day=100, val_end_day=140)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
