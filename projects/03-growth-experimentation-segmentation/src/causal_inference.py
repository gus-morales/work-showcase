"""
Difference-in-differences analysis of the collections-feature rollout.

Regions weren't randomized into the rollout (it went to half the regions
first based on business priority), so a simple before/after comparison
in the treated regions would confound the treatment effect with any
secular trend. Diff-in-differences nets out a shared time trend by
comparing the *change* in treated regions against the *change* in
control regions over the same window, and a pre-period trend check
(the "parallel trends" assumption) is run before trusting the estimate.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_RED, GREY, INK

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

ROLLOUT_DAY = 100


def parallel_trends_chart(df, source_note):
    daily = df.groupby(["day", "group"])["on_time_rate"].mean().reset_index()
    pivot = daily.pivot(index="day", columns="group", values="on_time_rate")
    pivot_smooth = pivot.rolling(7, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(pivot_smooth.index, pivot_smooth["control"] * 100, color=GREY, linewidth=1.8, label="Control regions")
    ax.plot(pivot_smooth.index, pivot_smooth["treated"] * 100, color=SLATE, linewidth=1.8, label="Treated regions")
    ax.axvline(ROLLOUT_DAY, color=MUTED_RED, linestyle="--", linewidth=1.3)
    ax.text(ROLLOUT_DAY + 2, ax.get_ylim()[0], "rollout", fontsize=9, color=MUTED_RED, va="bottom")
    style_ax(ax, title="Treated and control regions moved together before rollout, diverged after",
             subtitle="On-time repayment rate by day, 7-day rolling average",
             xlabel="Day", ylabel="On-time repayment rate (%)")
    ax.legend(loc="upper left", fontsize=9.5)
    savefig(fig, FIG_DIR / "did_parallel_trends.png", footnote=source_note)


def pretrend_placebo(df):
    """Check the parallel-trends assumption: in the pre-rollout window,
    the treated/control gap should not be trending, since neither group
    has been treated yet. A significant day x treated interaction here
    would be a warning sign that the two groups aren't a valid
    counterfactual for each other."""
    pre = df[~df.post_rollout].copy()
    pre["treated_flag"] = (pre.group == "treated").astype(int)
    model = smf.wls("on_time_rate ~ day * treated_flag", data=pre, weights=pre["n_customers"]).fit()
    coef = model.params["day:treated_flag"]
    pval = model.pvalues["day:treated_flag"]
    print(f"Pre-period trend difference (day x treated): {coef:.6f}/day, p={pval:.3f}")
    return coef, pval


def compute_diff_in_diff(df):
    """Pure computation half of the DiD estimate (no plotting), so it
    can be unit tested directly against a synthetic panel with a known
    injected effect."""
    d = df.copy()
    d["treated_flag"] = (d.group == "treated").astype(int)
    d["post_flag"] = d.post_rollout.astype(int)

    # Region fixed effects absorb the (time-invariant) treated_flag main
    # effect, and day fixed effects absorb the (region-invariant)
    # post_flag main effect, since post_flag is just a function of day.
    # Both main terms are therefore dropped as redundant; only the
    # treated_flag:post_flag interaction, which varies across both
    # region and time, is left to estimate.
    model = smf.wls(
        "on_time_rate ~ treated_flag:post_flag + C(region_id) + C(day)",
        data=d, weights=d["n_customers"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["region_id"]})

    did_coef = model.params["treated_flag:post_flag"]
    did_se = model.bse["treated_flag:post_flag"]
    did_p = model.pvalues["treated_flag:post_flag"]
    ci = (did_coef - 1.96 * did_se, did_coef + 1.96 * did_se)

    # Simple 2x2 mean version, for the intuitive cross-check alongside
    # the fixed-effects regression
    means = d.groupby(["group", "post_rollout"])["on_time_rate"].mean().unstack()
    simple_did = (means.loc["treated", True] - means.loc["treated", False]) - \
                 (means.loc["control", True] - means.loc["control", False])

    return {"did_coef": did_coef, "ci": ci, "p": did_p, "means": means, "simple_did": simple_did}


def diff_in_diff(df, source_note):
    result = compute_diff_in_diff(df)
    means, did_coef = result["means"], result["did_coef"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cats = ["Control\npre", "Control\npost", "Treated\npre", "Treated\npost"]
    vals = [means.loc["control", False] * 100, means.loc["control", True] * 100,
            means.loc["treated", False] * 100, means.loc["treated", True] * 100]
    colors = [GREY, GREY, MUTED_TEAL, SLATE]
    ax.bar(cats, vals, color=colors, width=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10, color=INK)
    style_ax(ax, title=f"Rollout raised on-time repayment by {did_coef*100:.1f}pp (DiD estimate)",
             subtitle="Region and day fixed-effects regression, clustered SE by region",
             ylabel="On-time repayment rate (%)")
    savefig(fig, FIG_DIR / "did_estimate.png", footnote=source_note)

    print(f"Simple 2x2 DiD: {result['simple_did']*100:.2f}pp")
    print(f"Fixed-effects DiD: {did_coef*100:.2f}pp, 95% CI [{result['ci'][0]*100:.2f}, {result['ci'][1]*100:.2f}]pp, "
          f"p={result['p']:.5f}")
    return {"did_coef": did_coef, "ci": result["ci"], "p": result["p"]}


def main():
    df = pd.read_csv(DATA_DIR / "regional_rollout.csv")
    n_regions = df.region_id.nunique()
    source_note = f"Source: synthetic BNPL regional panel · {n_regions} regions, {df.day.nunique()} days"

    parallel_trends_chart(df, source_note)
    pretrend_placebo(df)
    diff_in_diff(df, source_note)

    print("Wrote reports/figures/did_parallel_trends.png, did_estimate.png")


if __name__ == "__main__":
    main()
