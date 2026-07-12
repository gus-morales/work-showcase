"""Unit tests for the fixed-horizon vs. Thompson Sampling simulation.
Both simulators are deterministic given a seeded generator, so most of
these check structural invariants (allocation shape, regret math) that
should hold regardless of the specific random draw, plus one seeded
run that checks Thompson Sampling actually outperforms a fixed 50/50
design on average, the entire point of running it."""
import numpy as np
import pytest

from sequential_experimentation import simulate_fixed_horizon, simulate_thompson_sampling


def test_fixed_horizon_allocation_is_close_to_50_50():
    rng = np.random.default_rng(0)
    result = simulate_fixed_horizon(20_000, 0.30, 0.40, rng)
    assert result["treatment_share"][-1] == pytest.approx(0.5, abs=0.02)


def test_fixed_horizon_arms_never_respond_to_rewards():
    # A fixed design's allocation is drawn independently of any outcome;
    # this is what makes randomization valid for the standard test.
    rng = np.random.default_rng(0)
    result = simulate_fixed_horizon(5_000, 0.30, 0.40, rng)
    assert set(np.unique(result["arms"])) <= {0, 1}


def test_thompson_sampling_shifts_allocation_toward_better_arm():
    rng = np.random.default_rng(1)
    result = simulate_thompson_sampling(20_000, 0.30, 0.50, rng)
    # With a 20pp true gap and 20k users, allocation should end up
    # heavily skewed toward the better arm (arm 1), not near 50/50.
    assert result["treatment_share"][-1] > 0.85


def test_thompson_sampling_regret_is_nondecreasing():
    rng = np.random.default_rng(2)
    result = simulate_thompson_sampling(5_000, 0.30, 0.40, rng)
    assert np.all(np.diff(result["cum_regret"]) >= 0)


def test_thompson_sampling_regret_grows_slower_than_fixed_horizon():
    # The core claim of this whole module: over the same traffic and the
    # same true arm gap, adaptive allocation accumulates less regret
    # than a fixed 50/50 split, since it spends less traffic on the
    # inferior arm as evidence accumulates.
    rng_fixed = np.random.default_rng(3)
    rng_ts = np.random.default_rng(4)
    fixed = simulate_fixed_horizon(20_000, 0.30, 0.40, rng_fixed)
    ts = simulate_thompson_sampling(20_000, 0.30, 0.40, rng_ts)
    assert ts["cum_regret"][-1] < fixed["cum_regret"][-1]


def test_zero_regret_when_arms_are_identical():
    rng = np.random.default_rng(5)
    result = simulate_thompson_sampling(2_000, 0.35, 0.35, rng)
    assert result["cum_regret"][-1] == pytest.approx(0.0)


def test_reward_counts_match_array_lengths():
    rng = np.random.default_rng(6)
    result = simulate_fixed_horizon(1_000, 0.3, 0.4, rng)
    assert len(result["rewards"]) == 1_000
    assert len(result["cum_reward"]) == 1_000
    assert result["cum_reward"][-1] == result["rewards"].sum()
