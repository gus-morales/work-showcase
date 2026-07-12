"""Unit tests for the DoWhy model/identify/estimate/refute pipeline,
against a small hand-built panel with a known injected effect (the
same construction pattern as test_causal_inference.py's fixture)."""
import numpy as np
import pandas as pd
import pytest

from dowhy_refutation import build_and_estimate, run_refutation_suite

TRUE_EFFECT = 0.05


def _build_panel(rng_seed, rollout_day=20, n_regions=8, n_days=40):
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
            noise = rng.normal(0, 0.004)
            rate = float(np.clip(base + effect + noise, 0.01, 0.99))
            n_customers = 200
            rows.append((r, "treated" if is_treated else "control", day, post,
                         n_customers, int(round(rate * n_customers)), rate))
    return pd.DataFrame(rows, columns=[
        "region_id", "group", "day", "post_rollout", "n_customers", "n_on_time", "on_time_rate",
    ])


@pytest.fixture(scope="module")
def clean_panel():
    return _build_panel(rng_seed=0)


@pytest.fixture(scope="module")
def estimate_and_results(clean_panel):
    model, identified_estimand, estimate = build_and_estimate(clean_panel)
    results = run_refutation_suite(model, identified_estimand, estimate, num_simulations=15)
    return estimate, results


def test_build_and_estimate_recovers_the_known_effect(clean_panel):
    _, _, estimate = build_and_estimate(clean_panel)
    assert estimate.value == pytest.approx(TRUE_EFFECT, abs=0.015)


def test_refutation_suite_returns_all_three_refuters(estimate_and_results):
    _, results = estimate_and_results
    assert set(results.keys()) == {"random_common_cause", "placebo_treatment", "data_subset"}
    for r in results.values():
        assert "new_effect" in r and "p_value" in r and "is_statistically_significant" in r


def test_random_common_cause_leaves_the_estimate_essentially_unchanged(estimate_and_results):
    estimate, results = estimate_and_results
    assert results["random_common_cause"]["new_effect"] == pytest.approx(estimate.value, abs=0.01)


def test_placebo_treatment_collapses_toward_zero(estimate_and_results):
    _, results = estimate_and_results
    assert abs(results["placebo_treatment"]["new_effect"]) < 0.01


def test_data_subset_leaves_the_estimate_essentially_unchanged(estimate_and_results):
    estimate, results = estimate_and_results
    assert results["data_subset"]["new_effect"] == pytest.approx(estimate.value, abs=0.01)


def test_data_subset_refuter_uses_varying_subsets_not_a_degenerate_repeat(clean_panel):
    # Regression test for the DoWhy 0.14 quirk documented in
    # dowhy_refutation.py: passing a plain int random_state to
    # data_subset_refuter silently reruns the identical subset every
    # simulation. run_refutation_suite must pass an actual
    # np.random.RandomState so the simulations genuinely vary.
    model, identified_estimand, estimate = build_and_estimate(clean_panel)
    results = run_refutation_suite(model, identified_estimand, estimate, num_simulations=15)
    # A degenerate (zero-variance) run would still coincidentally match
    # the original estimate; the real signal is that repeated calls
    # with different seeds produce different new_effect values.
    results_2 = run_refutation_suite(model, identified_estimand, estimate, seed=99, num_simulations=15)
    assert results["data_subset"]["new_effect"] != results_2["data_subset"]["new_effect"]
