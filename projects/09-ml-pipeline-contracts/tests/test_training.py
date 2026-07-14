import numpy as np
import pandas as pd

from stages.training import _information_value, _psi, _reduce_features


def test_information_value_is_near_zero_for_an_uninformative_feature():
    rng = np.random.default_rng(0)
    n = 2000
    feature = pd.Series(rng.normal(size=n))          # unrelated to target
    target = pd.Series(rng.integers(0, 2, size=n))
    iv = _information_value(feature, target, bins=5)
    assert iv < 0.05


def test_information_value_is_high_for_a_strong_signal():
    rng = np.random.default_rng(0)
    n = 2000
    target = pd.Series(rng.integers(0, 2, size=n))
    feature = pd.Series(target * 5 + rng.normal(scale=0.3, size=n))  # strongly separates the classes
    iv = _information_value(feature, target, bins=5)
    assert iv > 0.5


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    train = pd.Series(rng.normal(size=2000))
    holdout = pd.Series(rng.normal(size=2000))
    assert _psi(train, holdout) < 0.05


def test_psi_is_high_for_a_shifted_distribution():
    rng = np.random.default_rng(0)
    train = pd.Series(rng.normal(loc=0, size=2000))
    holdout = pd.Series(rng.normal(loc=5, size=2000))  # badly shifted
    assert _psi(train, holdout) > 0.5


def test_reduce_features_never_returns_more_than_the_starting_feature_count():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    y = pd.Series((X["f0"] + rng.normal(scale=0.1, size=n) > 0).astype(int))
    split = n // 2
    model, best, history = _reduce_features(
        "classification", "pr_auc", "maximize",
        X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:],
    )
    assert len(best["features"]) <= 6
    assert len(history) >= 1
    assert history[0]["n_features"] == 6  # first step starts with everything


def test_reduce_features_stops_at_min_features_floor():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(4)})
    y = pd.Series((X["f0"] > 0).astype(int))
    split = n // 2
    model, best, history = _reduce_features(
        "classification", "pr_auc", "maximize",
        X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:],
    )
    assert min(h["n_features"] for h in history) >= 3  # MIN_FEATURES
