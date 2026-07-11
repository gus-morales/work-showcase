"""Unit tests for the diff-in-differences estimator and parallel-trends
placebo check, against small hand-built panels with a known injected
effect (or a known injected pre-trend violation) rather than the full
generated dataset, so the expected answer is known exactly."""
import numpy as np
import pandas as pd
import pytest

from causal_inference import pretrend_placebo, compute_diff_in_diff

TRUE_EFFECT = 0.05


def _build_panel(rng_seed, rollout_day=20, pretrend_slope=0.0, n_regions=8, n_days=40):
    rng = np.random.default_rng(rng_seed)
    regions = list(range(1, n_regions + 1))
    treated_regions = set(regions[: n_regions // 2])
    rows = []
    for r in regions:
        is_treated = r in treated_regions
        base = 0.55 if is_treated else 0.60
        for day in range(n_days):
            post = day >= rollout_day
            effect = TRUE_EFFECT if (is_treated and post) else 0.0
            pretrend = pretrend_slope * day if is_treated else 0.0
            noise = rng.normal(0, 0.004)
            rate = float(np.clip(base + effect + pretrend + noise, 0.01, 0.99))
            n_customers = 200
            rows.append((r, "treated" if is_treated else "control", day, post,
                         n_customers, int(round(rate * n_customers)), rate))
    return pd.DataFrame(rows, columns=[
        "region_id", "group", "day", "post_rollout", "n_customers", "n_on_time", "on_time_rate",
    ])


@pytest.fixture(scope="module")
def clean_panel():
    # No pre-trend violation, a known +5pp effect in treated regions after rollout.
    return _build_panel(rng_seed=0, pretrend_slope=0.0)


@pytest.fixture(scope="module")
def violated_panel():
    # Treated regions drift upward even before rollout - parallel trends is false here.
    return _build_panel(rng_seed=1, pretrend_slope=0.004)


def test_pretrend_placebo_passes_on_a_clean_panel(clean_panel):
    coef, pval = pretrend_placebo(clean_panel)
    assert abs(coef) < 0.001
    assert pval > 0.05


def test_pretrend_placebo_flags_a_violated_panel(violated_panel):
    coef, pval = pretrend_placebo(violated_panel)
    assert coef > 0.001
    assert pval < 0.05


def test_diff_in_diff_recovers_known_effect(clean_panel):
    result = compute_diff_in_diff(clean_panel)
    assert result["did_coef"] == pytest.approx(TRUE_EFFECT, abs=0.01)
    assert result["ci"][0] < TRUE_EFFECT < result["ci"][1]


def test_simple_2x2_and_fixed_effects_estimates_agree(clean_panel):
    result = compute_diff_in_diff(clean_panel)
    assert result["simple_did"] == pytest.approx(result["did_coef"], abs=0.01)
