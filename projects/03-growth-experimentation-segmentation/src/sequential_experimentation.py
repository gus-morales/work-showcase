"""
Same repayment-reminder test as experiment_design.py, run two different
ways over an identical traffic budget: the fixed 50/50 design actually
used (analyzed once, at the end, in experiment_design.py), and a
Thompson Sampling multi-armed bandit that reallocates traffic toward
the better-performing arm continuously as evidence accumulates.

This is a stylized two-armed comparison, not a re-run of the real
experiment: it simulates fresh Bernoulli outcomes from the two arms'
already-established conversion rates (34.3% control, 38.5% treatment,
see experiment_design.py and the README) rather than replaying the
actual per-user rows, since a bandit's allocation at step t depends on
the outcomes observed at steps 1..t-1, an ordering the already-collected
fixed-horizon data doesn't preserve. It also collapses the tenure-driven
heterogeneous treatment effect from uplift_modeling.py into two flat
rates; a contextual bandit that used tenure the way the CATE model does
would be the natural next step, not implemented here.

Run:
    python src/sequential_experimentation.py
Writes:
    reports/figures/bandit_traffic_allocation.png
    reports/figures/bandit_cumulative_regret.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

# Matches the realized control/treatment conversion rates from the actual
# experiment (experiment_design.py, README section 1).
P_CONTROL = 0.3425
P_TREATMENT = 0.3846
N_TOTAL = 40_000
SEED = 7


def simulate_fixed_horizon(n_total, p_control, p_treatment, rng):
    """A 50/50 randomized design: allocation never responds to outcomes,
    which is exactly what makes the standard two-proportion z-test in
    experiment_design.py valid."""
    arms = rng.integers(0, 2, size=n_total)  # 0 = control, 1 = treatment
    true_p = np.array([p_control, p_treatment])
    rewards = rng.binomial(1, true_p[arms])
    return _summarize(arms, rewards, true_p)


def simulate_thompson_sampling(n_total, p_control, p_treatment, rng):
    """Beta-Bernoulli Thompson Sampling: each step, draw a sample from
    each arm's current posterior over its conversion rate, play the arm
    with the higher sample, then update that arm's posterior with the
    observed outcome. Allocation drifts toward whichever arm looks
    better as the posteriors separate."""
    true_p = np.array([p_control, p_treatment])
    alpha = np.ones(2)
    beta = np.ones(2)
    arms = np.empty(n_total, dtype=int)
    rewards = np.empty(n_total, dtype=int)
    for t in range(n_total):
        sampled = rng.beta(alpha, beta)
        arm = int(np.argmax(sampled))
        reward = rng.binomial(1, true_p[arm])
        arms[t] = arm
        rewards[t] = reward
        alpha[arm] += reward
        beta[arm] += 1 - reward
    return _summarize(arms, rewards, true_p)


def _summarize(arms, rewards, true_p):
    p_best = true_p.max()
    per_user_regret = p_best - true_p[arms]
    n = len(arms)
    return {
        "arms": arms,
        "rewards": rewards,
        "cum_reward": np.cumsum(rewards),
        "cum_regret": np.cumsum(per_user_regret),
        "treatment_share": np.cumsum(arms) / np.arange(1, n + 1),
    }


def traffic_allocation_chart(fixed, ts, source_note):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, len(fixed["treatment_share"]) + 1)
    ax.plot(x, fixed["treatment_share"] * 100, color=GREY, linewidth=1.6, label="Fixed 50/50 design")
    ax.plot(x, ts["treatment_share"] * 100, color=SLATE, linewidth=1.8, label="Thompson Sampling")
    ax.axhline(50, color=GREY, linewidth=1, linestyle=":")
    style_ax(ax, title="Thompson Sampling shifts traffic toward the better arm as evidence accumulates",
             subtitle="Cumulative share of users allocated to the treatment arm",
             xlabel="User number (in traffic order)", ylabel="Share allocated to treatment (%)")
    ax.legend(fontsize=9.5, loc="lower right")
    savefig(fig, FIG_DIR / "bandit_traffic_allocation.png", footnote=source_note)


def cumulative_regret_chart(fixed, ts, source_note):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, len(fixed["cum_regret"]) + 1)
    ax.plot(x, fixed["cum_regret"], color=GREY, linewidth=1.6, label="Fixed 50/50 design")
    ax.plot(x, ts["cum_regret"], color=MUTED_TEAL, linewidth=1.8, label="Thompson Sampling")
    style_ax(ax, title="Adaptive allocation caps the cost of testing on the losing arm",
             subtitle="Cumulative regret: expected conversions forgone vs. always playing the better arm",
             xlabel="User number (in traffic order)", ylabel="Cumulative regret (expected conversions)")
    ax.legend(fontsize=9.5, loc="upper left")
    savefig(fig, FIG_DIR / "bandit_cumulative_regret.png", footnote=source_note)


def main():
    rng_fixed = np.random.default_rng(SEED)
    rng_ts = np.random.default_rng(SEED + 1)
    source_note = f"Source: simulated over the experiment's realized conversion rates (34.3% vs. 38.5%) · n = {N_TOTAL:,} users per design"

    fixed = simulate_fixed_horizon(N_TOTAL, P_CONTROL, P_TREATMENT, rng_fixed)
    ts = simulate_thompson_sampling(N_TOTAL, P_CONTROL, P_TREATMENT, rng_ts)

    additional_conversions = int(ts["cum_reward"][-1] - fixed["cum_reward"][-1])
    regret_reduction = 1 - ts["cum_regret"][-1] / fixed["cum_regret"][-1]
    final_treatment_share = ts["treatment_share"][-1]

    print(f"Fixed 50/50: {fixed['cum_reward'][-1]:,} conversions, "
          f"final cumulative regret {fixed['cum_regret'][-1]:.1f}")
    print(f"Thompson Sampling: {ts['cum_reward'][-1]:,} conversions, "
          f"final cumulative regret {ts['cum_regret'][-1]:.1f}, "
          f"{final_treatment_share:.1%} of traffic on treatment by the end")
    print(f"Additional conversions banked by Thompson Sampling over the same traffic: {additional_conversions:,}")
    print(f"Cumulative regret reduction: {regret_reduction:.1%}")

    traffic_allocation_chart(fixed, ts, source_note)
    cumulative_regret_chart(fixed, ts, source_note)

    summary = {
        "n_total": N_TOTAL,
        "p_control": P_CONTROL,
        "p_treatment": P_TREATMENT,
        "fixed_horizon": {
            "total_conversions": int(fixed["cum_reward"][-1]),
            "final_cumulative_regret": float(fixed["cum_regret"][-1]),
        },
        "thompson_sampling": {
            "total_conversions": int(ts["cum_reward"][-1]),
            "final_cumulative_regret": float(ts["cum_regret"][-1]),
            "final_treatment_share": float(final_treatment_share),
        },
        "additional_conversions_vs_fixed": additional_conversions,
        "cumulative_regret_reduction": float(regret_reduction),
    }
    with open(BASE / "reports" / "sequential_experimentation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote reports/figures/bandit_traffic_allocation.png, "
          "reports/figures/bandit_cumulative_regret.png, "
          "reports/sequential_experimentation_summary.json")


if __name__ == "__main__":
    main()
