"""Unit tests for the unsupervised anomaly-detection comparison. Checks
the sign convention on the anomaly score (a common source of silent
bugs with IsolationForest, since its raw score_samples output is higher
for normal points, the opposite of what "anomaly score" should mean)
and a directional sanity check that obvious outliers score as more
anomalous than routine points."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from anomaly_detection import anomaly_scores, fit_isolation_forest


@pytest.fixture
def clustered_data():
    # A tight cluster of "normal" points plus a handful of obvious
    # outliers, far outside the cluster on every dimension.
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0, scale=1, size=(300, 4))
    outliers = rng.normal(loc=25, scale=1, size=(10, 4))
    X = pd.DataFrame(np.vstack([normal, outliers]), columns=["a", "b", "c", "d"])
    is_outlier = np.array([False] * 300 + [True] * 10)
    return X, is_outlier


def test_anomaly_scores_are_higher_for_true_outliers(clustered_data):
    X, is_outlier = clustered_data
    iso = fit_isolation_forest(X)
    scores = anomaly_scores(iso, X)
    assert scores[is_outlier].mean() > scores[~is_outlier].mean()


def test_anomaly_scores_sign_convention_matches_sklearn_decision_function(clustered_data):
    # score_samples/decision_function are higher for inliers; anomaly_scores
    # negates that, so it should be exactly anti-correlated in rank.
    X, _ = clustered_data
    iso = IsolationForest(n_estimators=100, random_state=0).fit(X)
    raw = iso.score_samples(X)
    flipped = anomaly_scores(iso, X)
    assert np.corrcoef(raw, flipped)[0, 1] == pytest.approx(-1.0, abs=1e-9)


def test_fit_isolation_forest_returns_fitted_estimator(clustered_data):
    X, _ = clustered_data
    iso = fit_isolation_forest(X)
    # A fitted IsolationForest exposes estimators_; this fails before fit.
    assert hasattr(iso, "estimators_")
    assert len(iso.estimators_) > 0
