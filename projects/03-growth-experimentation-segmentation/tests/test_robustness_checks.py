"""Unit tests for the wild cluster bootstrap and Honest DiD robustness
checks, against small hand-built panels with a known injected effect
(and, for the Honest DiD case, a known injected pre-trend violation)
so the expected answer is known rather than eyeballed."""
import numpy as np
import pandas as pd
import pytest

from robustness_checks import (
    build_weekly_event_panel, wild_cluster_bootstrap_check, honest_did_check,
)

TRUE_EFFECT = 0.05
ROLLOUT_DAY = 98  # 14*7, a clean week boundary so no straddling week is dropped


def _build_panel(rng_seed, rollout_day=ROLLOUT_DAY, pretrend_slope=0.0, n_regions=8, n_days=182):
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
    return _build_panel(rng_seed=0, pretrend_slope=0.0)


@pytest.fixture(scope="module")
def violated_panel():
    # Treated regions drift upward even before rollout: a real pre-trend
    # violation, so Honest DiD should break down at a low M.
    return _build_panel(rng_seed=1, pretrend_slope=0.0015)


def test_weekly_panel_drops_the_straddling_week(clean_panel):
    panel, pre_weeks, post_weeks = build_weekly_event_panel(clean_panel, rollout_day=ROLLOUT_DAY)
    rollout_week = ROLLOUT_DAY // 7
    assert rollout_week not in pre_weeks
    assert rollout_week not in post_weeks
    assert set(panel["week_bin"].unique()) == set(pre_weeks) | set(post_weeks)


def test_weekly_panel_treated_flag_constant_per_region(clean_panel):
    panel, _, _ = build_weekly_event_panel(clean_panel, rollout_day=ROLLOUT_DAY)
    per_region_flags = panel.groupby("region_id")["treated_flag"].nunique()
    assert (per_region_flags == 1).all()


def test_wild_cluster_bootstrap_ci_brackets_the_known_effect(clean_panel):
    result = wild_cluster_bootstrap_check(clean_panel)
    assert result["coef"] == pytest.approx(TRUE_EFFECT, abs=0.01)
    assert result["bootstrap_ci"][0] < TRUE_EFFECT < result["bootstrap_ci"][1]


def test_wild_cluster_bootstrap_ci_close_to_analytical_ci(clean_panel):
    result = wild_cluster_bootstrap_check(clean_panel)
    boot_width = result["bootstrap_ci"][1] - result["bootstrap_ci"][0]
    analytical_width = result["analytical_ci"][1] - result["analytical_ci"][0]
    assert boot_width == pytest.approx(analytical_width, rel=0.5)


def test_honest_did_recovers_known_effect_on_a_clean_pretrend(clean_panel):
    result = honest_did_check(clean_panel, rollout_day=ROLLOUT_DAY)
    assert result["avg_att"] == pytest.approx(TRUE_EFFECT, abs=0.015)


def test_honest_did_widens_faster_on_a_violated_pretrend(clean_panel, violated_panel):
    # A real pre-trend violation means a bigger pre-period deviation to
    # scale the relative-magnitude restriction against, so the robust
    # interval should widen with M much faster than on a clean panel,
    # even if neither happens to cross zero within the tested grid.
    clean_df = honest_did_check(clean_panel, rollout_day=ROLLOUT_DAY)["sensitivity"].to_dataframe()
    violated_df = honest_did_check(violated_panel, rollout_day=ROLLOUT_DAY)["sensitivity"].to_dataframe()
    clean_width_at_half = clean_df.loc[clean_df["M"] == 0.5, "ci_ub"].iloc[0] - clean_df.loc[clean_df["M"] == 0.5, "ci_lb"].iloc[0]
    violated_width_at_half = violated_df.loc[violated_df["M"] == 0.5, "ci_ub"].iloc[0] - violated_df.loc[violated_df["M"] == 0.5, "ci_lb"].iloc[0]
    assert violated_width_at_half > clean_width_at_half * 2
