"""
A/B test on two reply-drafting prompt versions: a power analysis sized to
the smallest lift worth detecting, then a two-proportion test on the
actual result. Same structure as a UI experiment; the only difference is
the metric is judge-scored reply quality instead of a click or a
conversion event.

The absolute acceptable-rate here is inflated by the judge's generosity
bias (see eval_framework.py); that bias applies to both arms roughly
equally, so the *relative* lift between v1 and v2 is still a fair read,
even though neither arm's raw acceptable-rate should be quoted as a
trustworthy standalone number.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_AMBER, GREY

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

ALPHA = 0.05
POWER_TARGET = 0.80
BASELINE_RATE = 0.84  # observed v1 judge-acceptable rate, used to plan the follow-up test
PLANNED_MDE = 0.04


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def required_n_per_arm(mde, baseline=BASELINE_RATE, alpha=ALPHA, power=POWER_TARGET):
    analysis = NormalIndPower()
    effect_size = abs(cohens_h(baseline, baseline + mde))
    return analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="two-sided")


def compute_ab_result(df: pd.DataFrame) -> dict:
    """Pure computation: the two-proportion z-test and CI on the actual
    A/B data, independent of plotting."""
    counts = df.groupby("arm")["judge_acceptable"].agg(["sum", "count"])
    n_v1, n_v2 = counts.loc["v1_baseline", "count"], counts.loc["v2_revised", "count"]
    x_v1, x_v2 = counts.loc["v1_baseline", "sum"], counts.loc["v2_revised", "sum"]
    p_v1, p_v2 = x_v1 / n_v1, x_v2 / n_v2

    z_stat, p_value = proportions_ztest([x_v2, x_v1], [n_v2, n_v1], alternative="two-sided")
    ci_v1 = proportion_confint(x_v1, n_v1, alpha=ALPHA, method="wilson")
    ci_v2 = proportion_confint(x_v2, n_v2, alpha=ALPHA, method="wilson")

    lift_abs = p_v2 - p_v1
    se_diff = np.sqrt(p_v1 * (1 - p_v1) / n_v1 + p_v2 * (1 - p_v2) / n_v2)
    ci_diff = (lift_abs - 1.96 * se_diff, lift_abs + 1.96 * se_diff)

    return {
        "n_v1": int(n_v1), "n_v2": int(n_v2),
        "p_v1": float(p_v1), "p_v2": float(p_v2),
        "ci_v1": ci_v1, "ci_v2": ci_v2,
        "lift_abs": float(lift_abs), "ci_diff": ci_diff,
        "z_stat": float(z_stat), "p_value": float(p_value),
    }


def power_analysis(source_note):
    mde_range = np.arange(0.01, 0.09, 0.005)
    sample_sizes = [required_n_per_arm(m) for m in mde_range]
    planned_n = required_n_per_arm(PLANNED_MDE)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(mde_range * 100, sample_sizes, color=SLATE, linewidth=1.8)
    ax.scatter([PLANNED_MDE * 100], [planned_n], color=MUTED_AMBER, s=55, zorder=4)
    ax.annotate(f"planned MDE: {PLANNED_MDE*100:.0f}pp\nn = {planned_n:,.0f} per arm",
                xy=(PLANNED_MDE * 100, planned_n),
                xytext=(PLANNED_MDE * 100 + 1.0, planned_n + 3000), fontsize=9.5, color=MUTED_AMBER)
    style_ax(ax, title="Smaller quality lifts require rapidly more traffic to detect",
             subtitle=f"Required sample size per arm, baseline acceptable-rate {BASELINE_RATE:.0%}, "
                      f"alpha={ALPHA}, power={POWER_TARGET:.0%}",
             xlabel="Minimum detectable lift (absolute, percentage points)", ylabel="Required sample size per arm")
    savefig(fig, FIG_DIR / "ab_power_analysis.png", footnote=source_note)
    print(f"Planned test: MDE={PLANNED_MDE*100:.0f}pp, required n/arm={planned_n:,.0f}")
    return planned_n


def ab_test(df, source_note):
    result = compute_ab_result(df)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    arms = ["v1 (baseline)", "v2 (revised)"]
    rates = [result["p_v1"] * 100, result["p_v2"] * 100]
    errs = [(result["p_v1"] - result["ci_v1"][0]) * 100, (result["p_v2"] - result["ci_v2"][0]) * 100]
    ax.bar(arms, rates, yerr=[errs, errs], color=[SLATE, MUTED_TEAL], width=0.5, zorder=3,
           error_kw={"ecolor": GREY, "elinewidth": 1.2, "capsize": 4})
    for i, v in enumerate(rates):
        ax.text(i, v + max(errs) + 0.5, f"{v:.1f}%", ha="center", fontsize=10.5, color="#333")
    style_ax(ax, title=f"v2 lifts judge-acceptable rate by {result['lift_abs']*100:.1f}pp (p={result['p_value']:.4f})",
             subtitle="Judge-scored acceptable rate, with 95% Wilson confidence intervals",
             ylabel="Acceptable rate (%)")
    savefig(fig, FIG_DIR / "ab_test_result.png", footnote=source_note)

    print(f"v1: {result['p_v1']:.4f} ({result['n_v1']} tickets)  v2: {result['p_v2']:.4f} ({result['n_v2']} tickets)")
    print(f"Absolute lift: {result['lift_abs']*100:.2f}pp, 95% CI "
          f"[{result['ci_diff'][0]*100:.2f}, {result['ci_diff'][1]*100:.2f}]pp")
    print(f"z={result['z_stat']:.3f}, p={result['p_value']:.5f}")
    return result


def main():
    df = pd.read_csv(DATA_DIR / "ab_test_results.csv")
    source_note = f"Source: synthetic A/B test data · n = {len(df):,} tickets"
    power_analysis(source_note)
    ab_test(df, source_note)
    print("Wrote reports/figures/ab_power_analysis.png, ab_test_result.png")


if __name__ == "__main__":
    main()
