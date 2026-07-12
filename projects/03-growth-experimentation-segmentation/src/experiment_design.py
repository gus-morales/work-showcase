"""
Experiment design and analysis for the repayment-reminder A/B test:

1. Power analysis - the sample size that should have been planned for,
   given a target minimum detectable effect (MDE), before the test ran.
2. Standard analysis - a two-proportion z-test on the primary conversion
   metric.
3. CUPED - a pre-period revenue covariate used to reduce variance on the
   secondary continuous revenue metric, tightening the confidence
   interval without spending more traffic.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy import stats

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_AMBER, GREY, INK

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

ALPHA = 0.05
POWER_TARGET = 0.80
BASELINE_RATE = 0.34


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def power_analysis(source_note):
    analysis = NormalIndPower()
    mde_range = np.arange(0.01, 0.09, 0.005)  # absolute lift in conversion rate
    sample_sizes = []
    for mde in mde_range:
        effect_size = abs(cohens_h(BASELINE_RATE, BASELINE_RATE + mde))
        n = analysis.solve_power(effect_size=effect_size, alpha=ALPHA, power=POWER_TARGET, alternative="two-sided")
        sample_sizes.append(n)
    sample_sizes = np.array(sample_sizes)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(mde_range * 100, sample_sizes, color=SLATE, linewidth=1.8)
    planned_mde = 0.035
    planned_n = analysis.solve_power(effect_size=abs(cohens_h(BASELINE_RATE, BASELINE_RATE + planned_mde)),
                                      alpha=ALPHA, power=POWER_TARGET, alternative="two-sided")
    ax.scatter([planned_mde * 100], [planned_n], color=MUTED_AMBER, s=55, zorder=4)
    ax.annotate(f"planned MDE: 3.5pp\nn = {planned_n:,.0f} per arm", xy=(planned_mde * 100, planned_n),
                xytext=(planned_mde * 100 + 1.2, planned_n + 4000), fontsize=9.5, color=MUTED_AMBER)
    style_ax(ax, title="Smaller effects require rapidly more traffic to detect",
             subtitle=f"Required sample size per arm, baseline conversion {BASELINE_RATE:.0%}, "
                      f"alpha={ALPHA}, power={POWER_TARGET:.0%}",
             xlabel="Minimum detectable effect (absolute, percentage points)", ylabel="Required sample size per arm")
    savefig(fig, FIG_DIR / "power_analysis.png", footnote=source_note)

    print(f"Planned test: MDE=3.5pp, required n/arm={planned_n:,.0f}")
    return planned_n


def analyze_conversion(df, source_note):
    counts = df.groupby("arm")["converted_post_14d"].agg(["sum", "count"])
    n_control, n_treat = counts.loc["control", "count"], counts.loc["treatment", "count"]
    x_control, x_treat = counts.loc["control", "sum"], counts.loc["treatment", "sum"]
    p_control, p_treat = x_control / n_control, x_treat / n_treat

    z_stat, p_value = proportions_ztest([x_treat, x_control], [n_treat, n_control], alternative="two-sided")
    ci_control = proportion_confint(x_control, n_control, alpha=ALPHA, method="wilson")
    ci_treat = proportion_confint(x_treat, n_treat, alpha=ALPHA, method="wilson")

    lift_abs = p_treat - p_control
    lift_rel = lift_abs / p_control
    se_diff = np.sqrt(p_control * (1 - p_control) / n_control + p_treat * (1 - p_treat) / n_treat)
    ci_diff = (lift_abs - 1.96 * se_diff, lift_abs + 1.96 * se_diff)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    arms = ["Control", "Treatment"]
    rates = [p_control * 100, p_treat * 100]
    errs = [(p_control - ci_control[0]) * 100, (p_treat - ci_treat[0]) * 100]
    ax.bar(arms, rates, yerr=[errs, errs], color=[SLATE, MUTED_TEAL], width=0.5, zorder=3,
           error_kw={"ecolor": GREY, "elinewidth": 1.2, "capsize": 4})
    for i, v in enumerate(rates):
        ax.text(i, v + max(errs) + 0.5, f"{v:.1f}%", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title=f"Treatment lifts on-time conversion by {lift_abs*100:.1f}pp (p={p_value:.4f})",
             subtitle="14-day post-exposure conversion rate, with 95% Wilson confidence intervals",
             ylabel="Conversion rate (%)")
    savefig(fig, FIG_DIR / "ab_conversion_result.png", footnote=source_note)

    print(f"Control: {p_control:.4f} ({x_control}/{n_control})  Treatment: {p_treat:.4f} ({x_treat}/{n_treat})")
    print(f"Absolute lift: {lift_abs*100:.2f}pp ({lift_rel:.1%} relative), 95% CI [{ci_diff[0]*100:.2f}, {ci_diff[1]*100:.2f}]pp")
    print(f"z={z_stat:.3f}, p={p_value:.5f}")
    return {"lift_abs": lift_abs, "p_value": p_value, "ci_diff": ci_diff}


def analyze_cuped(df, source_note):
    """CUPED: adjust the revenue outcome using the pre-period revenue
    covariate to strip out variance unrelated to treatment, tightening
    the confidence interval on the estimated lift without adding
    traffic. Theta is estimated pooled across arms (standard practice,
    since it should not itself depend on treatment)."""
    y = df["revenue_post_14d_usd"].values
    x = df["revenue_pre_30d_usd"].values
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    df = df.copy()
    df["revenue_adj"] = y_adj

    def arm_stats(col):
        c = df[df.arm == "control"][col]
        t = df[df.arm == "treatment"][col]
        diff = t.mean() - c.mean()
        se = np.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
        ci = (diff - 1.96 * se, diff + 1.96 * se)
        t_stat, p_val = stats.ttest_ind(t, c, equal_var=False)
        return diff, se, ci, p_val

    diff_raw, se_raw, ci_raw, p_raw = arm_stats("revenue_post_14d_usd")
    diff_adj, se_adj, ci_adj, p_adj = arm_stats("revenue_adj")
    variance_reduction = 1 - (se_adj / se_raw) ** 2
    ci_width_reduction = 1 - se_adj / se_raw

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ["Standard\n(raw revenue)", "CUPED-adjusted"]
    diffs = [diff_raw, diff_adj]
    errs = [1.96 * se_raw, 1.96 * se_adj]
    colors = [GREY, SLATE]
    ax.bar(labels, diffs, yerr=errs, color=colors, width=0.45, zorder=3,
           error_kw={"ecolor": INK, "elinewidth": 1.3, "capsize": 5})
    ax.axhline(0, color=GREY, linewidth=1)
    ylim_max = max(d + e for d, e in zip(diffs, errs)) * 1.35
    ax.set_ylim(0, ylim_max)
    for i, v in enumerate(diffs):
        ax.text(i, v + errs[i] + ylim_max * 0.03, f"+{v:.1f} USD", ha="center", fontsize=10, color=INK)
    style_ax(ax, title=f"CUPED narrows the confidence interval by {ci_width_reduction:.0%} on the same data",
             subtitle="Estimated treatment lift in 14-day revenue per user, 95% CI",
             ylabel="Estimated lift (USD)")
    savefig(fig, FIG_DIR / "cuped_variance_reduction.png", footnote=source_note)

    print(f"Raw lift: {diff_raw:.2f} USD, 95% CI [{ci_raw[0]:.2f}, {ci_raw[1]:.2f}], p={p_raw:.4f}")
    print(f"CUPED lift: {diff_adj:.2f} USD, 95% CI [{ci_adj[0]:.2f}, {ci_adj[1]:.2f}], p={p_adj:.4f}")
    print(f"Variance reduction from CUPED: {variance_reduction:.1%}")
    return {"diff_raw": diff_raw, "diff_adj": diff_adj, "variance_reduction": variance_reduction}


def main():
    df = pd.read_csv(DATA_DIR / "experiment_users.csv")
    n_users = len(df)
    source_note = f"Source: synthetic BNPL experiment data · n = {n_users:,} users"

    power_analysis(source_note)
    analyze_conversion(df, source_note)
    analyze_cuped(df, source_note)

    print("Wrote reports/figures/power_analysis.png, ab_conversion_result.png, cuped_variance_reduction.png")


if __name__ == "__main__":
    main()
