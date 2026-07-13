"""Unit tests for the feature pipeline and the temporal split, on tiny
hand-built fixtures instead of the full generated dataset."""
import pandas as pd
import pytest

from features import build_feature_pipeline, temporal_split, CATEGORICAL_COLS, RAW_FEATURE_COLS


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "engine_hours_at_obs": [1000, 40000],
        "equipment_age_years_at_obs": [1.0, 12.0],
        "vibration_rms_mm_s": [2.0, 40.0],  # second row exercises the winsorizer's upper cap
        "oil_pressure_psi": [58.0, 40.0],
        "coolant_temp_c": [86.0, 95.0],
        "brake_wear_pct": [10.0, 60.0],
        "fault_codes_7d": [0, 3],
        "days_since_last_pm": [5, 50],
        "payload_avg_tons": [200.0, 380.0],
        "ambient_temp_c": [24.0, 31.0],
        "truck_class": ["standard_class_200t", "ultra_class_400t"],
        "site": ["north_pit", "south_pit"],
    })


def test_pipeline_one_hot_encodes_categoricals(sample_df):
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(sample_df[RAW_FEATURE_COLS])
    for col in CATEGORICAL_COLS:
        assert not any(c == col for c in X.columns)
    assert any(c.startswith("truck_class_") for c in X.columns)
    assert any(c.startswith("site_") for c in X.columns)
    assert X.shape[0] == len(sample_df)


def test_pipeline_winsorizes_extreme_vibration():
    # A training split with a tight, low-variance range; a wildly larger
    # test-split value should get capped rather than passed through raw.
    train = pd.DataFrame({
        "engine_hours_at_obs": [1000, 1100, 1050, 950, 1020, 980],
        "equipment_age_years_at_obs": [1.0] * 6,
        "vibration_rms_mm_s": [2.0, 2.1, 2.2, 1.9, 2.0, 2.1],
        "oil_pressure_psi": [58.0, 57.5, 58.5, 57.8, 58.2, 58.1],
        "coolant_temp_c": [86.0, 85.5, 86.5, 85.8, 86.2, 86.1],
        "brake_wear_pct": [10.0, 9.5, 10.5, 9.8, 10.2, 10.1],
        "fault_codes_7d": [0] * 6,
        "days_since_last_pm": [5] * 6,
        "payload_avg_tons": [200.0, 199.0, 201.0, 199.5, 200.5, 200.2],
        "ambient_temp_c": [24.0] * 6,
        "truck_class": ["standard_class_200t"] * 6,
        "site": ["north_pit"] * 6,
    })
    test = train.iloc[:1].copy()
    test["vibration_rms_mm_s"] = 18.0

    pipeline = build_feature_pipeline()
    pipeline.fit(train[RAW_FEATURE_COLS])
    X_test = pipeline.transform(test[RAW_FEATURE_COLS])
    assert X_test.loc[0, "vibration_rms_mm_s"] < 18.0


def test_pipeline_has_no_missing_values(sample_df):
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(sample_df[RAW_FEATURE_COLS])
    assert not X.isnull().any().any()


def test_temporal_split_partitions_are_disjoint_and_complete():
    df = pd.DataFrame({
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(range(600), unit="D"),
        "value": range(600),
    })
    train, val, test = temporal_split(df, train_end_day=400, val_end_day=500)
    assert len(train) + len(val) + len(test) == len(df)
    assert set(train["value"]) & set(val["value"]) == set()
    assert set(val["value"]) & set(test["value"]) == set()


def test_temporal_split_train_precedes_val_precedes_test():
    df = pd.DataFrame({
        "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(range(600), unit="D"),
        "value": range(600),
    })
    train, val, test = temporal_split(df, train_end_day=400, val_end_day=500)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
