"""Unit tests for the cumulative-gains computation and the one-hot
feature builder: pure functions tested on small hand-built cases,
independent of the actual trained model."""
import numpy as np
import pandas as pd
import pytest

from propensity_model import build_features, cumulative_gains, FEATURE_NUM, FEATURE_CAT


def test_gains_endpoints_are_zero_and_one():
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    y_score = np.linspace(1, 0, 10)
    pop_share, capture_share = cumulative_gains(y_true, y_score, n_bins=5)
    assert pop_share[0] == 0.0 and capture_share[0] == 0.0
    assert pop_share[-1] == pytest.approx(1.0)
    assert capture_share[-1] == pytest.approx(1.0)


def test_perfect_ranking_captures_all_positives_in_the_first_bin_that_fits_them():
    # 10 customers, exactly the top 2 are true positives, scored highest.
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.95, 0.90, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    pop_share, capture_share = cumulative_gains(y_true, y_score, n_bins=5)
    # First bin = top 20% = exactly the 2 positives -> 100% captured already.
    assert pop_share[1] == pytest.approx(0.2)
    assert capture_share[1] == pytest.approx(1.0)


def test_worst_case_ranking_delays_capture_to_the_final_bins():
    # The 2 true positives are scored lowest, so early bins should
    # capture none of them.
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    pop_share, capture_share = cumulative_gains(y_true, y_score, n_bins=5)
    assert capture_share[1] == pytest.approx(0.0)
    assert capture_share[2] == pytest.approx(0.0)


def test_capture_share_is_monotonically_nondecreasing():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.random(200)
    _, capture_share = cumulative_gains(y_true, y_score, n_bins=10)
    assert (np.diff(capture_share) >= 0).all()


def test_build_features_one_hot_encodes_categoricals_and_keeps_numeric():
    df = pd.DataFrame({
        "tenure_days": [100, 200], "lifetime_orders": [5, 10], "recency_days": [10, 90],
        "frequency_90d": [3, 1], "monetary_90d": [50.0, 20.0], "category_diversity": [2, 1],
        "decline_rate": [0.1, 0.2], "segment": ["Lapsed", "Dormant"], "primary_channel": ["ios", "web"],
    })
    X = build_features(df)
    for col in FEATURE_NUM:
        assert col in X.columns
    for col in FEATURE_CAT:
        assert not any(c == col for c in X.columns)  # replaced by dummies, not kept as-is
    assert "segment_Lapsed" in X.columns
    assert "primary_channel_ios" in X.columns
    assert X["tenure_days"].tolist() == [100, 200]
