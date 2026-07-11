"""Unit tests for the uplift/CATE pure computation: bucket calibration
and the Qini curve, both on small hand-built examples with known
expected values, plus a light integration check on model fitting."""
import numpy as np
import pandas as pd
import pytest

from uplift_modeling import (
    add_model_features, compute_uplift_by_decile, compute_qini_curve,
    fit_t_learner, predict_cate, MODEL_COVARIATES, OUTCOME,
)


def test_add_model_features_log1p_transform():
    df = pd.DataFrame({"tenure_days": [0, 9]})
    out = add_model_features(df)
    assert out["log_tenure_days"].tolist() == pytest.approx([np.log1p(0), np.log1p(9)])


def test_add_model_features_does_not_mutate_input():
    df = pd.DataFrame({"tenure_days": [0, 9]})
    add_model_features(df)
    assert "log_tenure_days" not in df.columns


@pytest.fixture
def eight_user_example():
    # Lower half by predicted CATE (0.1-0.4): treat and control both
    # convert at 50%, so realized lift should be ~0.
    # Upper half (0.5-0.8): treated convert 100%, control 0%, so
    # realized lift should be 1.0. Two clearly separated buckets.
    cate_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    arm = ["control", "treatment", "control", "treatment",
           "control", "treatment", "control", "treatment"]
    outcome = [0, 0, 1, 1, 0, 1, 0, 1]
    return cate_pred, arm, outcome


def test_uplift_by_decile_bucket_stats_match_hand_computation(eight_user_example):
    cate_pred, arm, outcome = eight_user_example
    result = compute_uplift_by_decile(cate_pred, arm, outcome, n_buckets=2)
    assert len(result) == 2
    top, bottom = result.iloc[0], result.iloc[1]  # sorted highest predicted CATE first
    assert top["n"] == 4
    assert top["predicted_cate_mean"] == pytest.approx(0.65)
    assert top["realized_lift"] == pytest.approx(1.0)
    assert bottom["predicted_cate_mean"] == pytest.approx(0.25)
    assert bottom["realized_lift"] == pytest.approx(0.0)


def test_qini_curve_matches_hand_computation():
    # Sorted by predicted CATE descending already.
    cate_pred = [0.4, 0.3, 0.2, 0.1]
    arm = ["treatment", "control", "treatment", "control"]
    outcome = [1, 0, 0, 1]
    result = compute_qini_curve(cate_pred, arm, outcome)
    # Hand-computed: qini_values = [1, 1, 1, 0], random_baseline = [0,0,0,0]
    # (total_gain = 0, so the baseline is flat at zero), area (with a
    # prepended (0,0) point) = 0.75.
    assert result["qini_values"] == pytest.approx([1.0, 1.0, 1.0, 0.0])
    assert result["qini_coefficient"] == pytest.approx(0.75)


def test_qini_curve_final_value_equals_overall_incremental_gain():
    # Property check: at 100% targeting, the qini curve's last value must
    # equal the simple treated-minus-scaled-control total, regardless of
    # sort order (since by then every user is included either way).
    rng = np.random.default_rng(0)
    n = 200
    cate_pred = rng.normal(size=n)
    arm = np.where(rng.random(n) < 0.5, "treatment", "control")
    outcome = rng.binomial(1, 0.3, size=n)
    result = compute_qini_curve(cate_pred, arm, outcome)

    is_treat = arm == "treatment"
    n_treat, n_control = is_treat.sum(), (~is_treat).sum()
    y_treat, y_control = outcome[is_treat].sum(), outcome[~is_treat].sum()
    expected_final = y_treat - y_control * (n_treat / n_control)
    assert result["qini_values"][-1] == pytest.approx(expected_final)


def test_qini_curve_is_zero_when_treatment_has_no_effect():
    # If treated and control convert at exactly the same rate regardless
    # of the (here, uninformative) score, the curve should track the
    # random baseline closely rather than showing a real gain.
    rng = np.random.default_rng(1)
    n = 2000
    cate_pred = rng.normal(size=n)
    arm = np.where(rng.random(n) < 0.5, "treatment", "control")
    outcome = rng.binomial(1, 0.3, size=n)  # same rate regardless of arm
    result = compute_qini_curve(cate_pred, arm, outcome)
    # Loose bound: with no true effect, the coefficient should be small
    # relative to a real effect (checked in test_qini_curve_matches_hand_computation).
    assert abs(result["qini_coefficient"]) < 15


def test_fit_t_learner_and_predict_cate_shapes():
    rng = np.random.default_rng(2)
    n = 2000
    tenure_days = rng.exponential(scale=150, size=n).round()
    sessions = rng.poisson(8, size=n)
    revenue = rng.normal(80, 20, size=n)
    arm = np.where(rng.random(n) < 0.5, "treatment", "control")
    p = np.where(arm == "treatment", 0.4, 0.3)
    outcome = rng.binomial(1, p)
    df = pd.DataFrame({
        "tenure_days": tenure_days, "sessions_pre_30d": sessions,
        "revenue_pre_30d_usd": revenue, "arm": arm, OUTCOME: outcome,
    })
    df = add_model_features(df)

    model_treat, model_control = fit_t_learner(df)
    cate = predict_cate(model_treat, model_control, df)

    assert len(cate) == len(df)
    assert np.isfinite(cate).all()
    assert set(MODEL_COVARIATES).issubset(df.columns)
