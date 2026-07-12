"""Unit tests for the survival-analysis module: covariate construction
and a directional sanity check on the fitted Cox model, using a small
hand-built dataset instead of the full generated portfolio so the
Cox fit itself stays fast and deterministic."""
import numpy as np
import pandas as pd
import pytest

from survival_analysis import build_cox_covariates, fit_cox_model


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(0)
    n = 400
    # Two clean groups: "risky" loans fail fast and almost always fail;
    # "safe" loans rarely fail and, when they do, fail slowly. A Cox fit
    # on this should recover a hazard ratio well above 1 for the risky
    # group with no ambiguity.
    group = np.where(np.arange(n) < n // 2, "risky", "safe")
    event = np.where(group == "risky", rng.binomial(1, 0.8, n), rng.binomial(1, 0.1, n))
    time = np.where(
        group == "risky",
        rng.uniform(5, 40, n),
        rng.uniform(150, 300, n),
    )
    employment_type = np.where(
        group == "risky",
        rng.choice(["informal", "gig_economy"], n, p=[0.7, 0.3]),
        rng.choice(["salaried", "self_employed"], n, p=[0.7, 0.3]),
    )
    return pd.DataFrame({
        "employment_type": employment_type,
        # A little unrelated variation in the other categoricals so none
        # of the dummy columns are degenerate (all-zero), which is what a
        # real portfolio would look like and what the Cox fit needs to
        # converge cleanly.
        "merchant_category": rng.choice(["other", "electronics", "travel"], n, p=[0.7, 0.2, 0.1]),
        "avg_prior_repayment_delay_days": rng.uniform(0, 10, n),
        "num_active_loans_elsewhere": rng.integers(0, 3, n),
        "monthly_income_usd": rng.uniform(1000, 4000, n),
        "loan_amount_usd": rng.uniform(500, 3000, n),
        "num_installments": rng.choice([3, 6, 12], n),
        "down_payment_ratio": rng.uniform(0, 0.3, n),
        "tenure_months_platform": rng.uniform(0, 24, n),
        "credit_bureau_score_filled": np.where(
            group == "risky", rng.normal(550, 40, n), rng.normal(720, 40, n)
        ).clip(300, 850),
        "bureau_score_missing": rng.binomial(1, 0.05, n),
        "time_to_30dpd_days": time,
        "event_observed": event,
    })


def test_build_cox_covariates_shape_and_dummy_encoding(toy_df):
    cov = build_cox_covariates(toy_df)
    assert "employment_informal" in cov.columns
    assert "employment_salaried" not in cov.columns  # reference level, not a column
    assert set(cov["employment_informal"].unique()) <= {0, 1}
    assert len(cov) == len(toy_df)


def test_cox_model_recovers_higher_hazard_for_risky_group(toy_df):
    cov = build_cox_covariates(toy_df)
    cph = fit_cox_model(cov)
    # employment and bureau score are both correlated with the same risky/
    # safe split here (as they would be in the real generator too), so
    # only the sign is checked; the two covariates share credit for the
    # same underlying signal and neither one alone needs to clear
    # significance for the model to be directionally right.
    assert cph.summary.loc["employment_informal", "exp(coef)"] > 1
    assert cph.summary.loc["credit_bureau_score_filled", "exp(coef)"] < 1


def test_concordance_index_is_well_above_chance(toy_df):
    cov = build_cox_covariates(toy_df)
    cph = fit_cox_model(cov)
    # 0.5 is chance-level discrimination; this dataset is constructed to
    # be an easy separation, so the fitted model should do much better.
    assert cph.concordance_index_ > 0.75
