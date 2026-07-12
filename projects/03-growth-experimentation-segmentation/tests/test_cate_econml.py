"""Unit tests for the CausalForestDML comparison: shape/sanity checks
on model fitting and predictions, plus the core empirical claim this
module exists to check, that a DML-based estimator has a higher Qini
coefficient than the T-learner on the same noisy data."""
import numpy as np
import pandas as pd
import pytest

from cate_econml import fit_causal_forest_dml, predict_cate_econml, econml_comparison
from uplift_modeling import add_model_features, OUTCOME, MODEL_COVARIATES


def _build_experiment_panel(rng_seed, n=3000):
    rng = np.random.default_rng(rng_seed)
    tenure_days = np.clip(rng.exponential(scale=180, size=n), 0, 900).round()
    sessions = rng.poisson(8, size=n)
    revenue = rng.normal(80, 20, size=n)
    arm = np.where(rng.random(n) < 0.5, "treatment", "control")
    # Same heterogeneous-effect shape as generate_data.py: decays with tenure.
    lift = np.where(arm == "treatment", np.clip(0.005 + 0.11 * np.exp(-tenure_days / 100), 0.0, 0.16), 0.0)
    p = np.clip(0.34 + lift, 0.01, 0.99)
    outcome = rng.binomial(1, p)
    df = pd.DataFrame({
        "tenure_days": tenure_days, "sessions_pre_30d": sessions,
        "revenue_pre_30d_usd": revenue, "arm": arm, OUTCOME: outcome,
    })
    return add_model_features(df)


@pytest.fixture(scope="module")
def experiment_panel():
    return _build_experiment_panel(rng_seed=4)


def test_fit_causal_forest_dml_and_predict_shapes(experiment_panel):
    est = fit_causal_forest_dml(experiment_panel)
    cate = predict_cate_econml(est, experiment_panel)
    assert len(cate) == len(experiment_panel)
    assert np.isfinite(cate).all()


def test_predict_cate_econml_accepts_a_dataframe_missing_extra_columns(experiment_panel):
    # predict_cate_econml should only need MODEL_COVARIATES present, not
    # every column the training frame happened to have.
    est = fit_causal_forest_dml(experiment_panel)
    minimal = experiment_panel[MODEL_COVARIATES].copy()
    cate = predict_cate_econml(est, minimal)
    assert len(cate) == len(minimal)


def test_econml_comparison_reports_higher_qini_than_tlearner(tmp_path, monkeypatch):
    import cate_econml
    monkeypatch.setattr(cate_econml, "FIG_DIR", tmp_path)
    df = _build_experiment_panel(rng_seed=5, n=4000)
    # add_model_features already applied by the fixture builder; econml_comparison
    # re-applies it internally (idempotent, just adds log_tenure_days again).
    result = econml_comparison(df, source_note="test")
    assert result["qini_econml"] > result["qini_tlearner"]


def test_ate_interval_brackets_the_ate_point(tmp_path, monkeypatch):
    import cate_econml
    monkeypatch.setattr(cate_econml, "FIG_DIR", tmp_path)
    df = _build_experiment_panel(rng_seed=6, n=4000)
    result = econml_comparison(df, source_note="test")
    lb, ub = result["ate_ci"]
    assert lb < result["ate_point"] < ub
