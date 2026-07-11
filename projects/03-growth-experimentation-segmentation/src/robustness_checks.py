"""
Robustness checks on the regional-rollout DiD estimate, using the
diff-diff package (github.com/igerber/diff-diff) for two things the
hand-rolled fixed-effects regression in causal_inference.py doesn't
cover on its own:

1. Wild cluster bootstrap inference. The primary DiD regression clusters
   standard errors by region (40 clusters, 20 treated), on the edge of
   "few clusters" territory where asymptotic cluster-robust SEs can be
   unreliable. The wild cluster bootstrap (Cameron, Gelbach & Miller
   2008) is run here as a finite-sample-robust cross-check on the same
   point estimate, not a replacement for it.

2. Honest DiD (Rambachan & Roth 2023). The pre-trends placebo check in
   causal_inference.py tests whether the pre-period trend difference is
   statistically distinguishable from zero, but says nothing about how
   large an undetected violation could be while still leaving the
   conclusion intact. Honest DiD reports how robust the post-period
   estimate is to potential parallel-trends violations up to some
   multiple (M) of the largest pre-period deviation actually observed,
   and the breakdown value of M at which the estimate stops being
   statistically distinguishable from zero.
"""
from pathlib import Path

import diff_diff as dd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

ROLLOUT_DAY = 100
SEED = 13


def wild_cluster_bootstrap_check(df):
    """Refits the same region+day fixed-effects WLS regression as
    causal_inference.py, then runs a wild cluster bootstrap on the
    treated_flag:post_flag coefficient. Uses the WLS-to-OLS transform
    (regressing sqrt(weight)*y on sqrt(weight)*X, which statsmodels
    already exposes as wexog/wendog) so the bootstrap runs on exactly
    the coefficient the primary model reports, not an unweighted
    approximation of it."""
    d = df.copy()
    d["treated_flag"] = (d.group == "treated").astype(int)
    d["post_flag"] = d.post_rollout.astype(int)

    model = smf.wls(
        "on_time_rate ~ treated_flag:post_flag + C(region_id) + C(day)",
        data=d, weights=d["n_customers"],
    ).fit(cov_type="cluster", cov_kwds={"groups": d["region_id"]})

    coef_index = model.model.exog_names.index("treated_flag:post_flag")
    ci = model.conf_int().iloc[coef_index]
    boot = dd.wild_bootstrap_se(
        X=model.model.wexog, y=model.model.wendog, residuals=model.resid,
        cluster_ids=d["region_id"].values, coefficient_index=coef_index,
        n_bootstrap=999, weight_type="rademacher", seed=SEED,
    )
    return {
        "coef": model.params.iloc[coef_index],
        "analytical_ci": (ci.iloc[0], ci.iloc[1]),
        "bootstrap_ci": (boot.ci_lower, boot.ci_upper),
        "bootstrap_p": boot.p_value,
        "n_clusters": boot.n_clusters,
    }


def build_weekly_event_panel(df, rollout_day=ROLLOUT_DAY):
    """Bins day into 7-day periods for the event-study specification
    Honest DiD needs: per-period pre/post treatment effects instead of a
    single pooled post_flag. The week straddling the rollout day is
    dropped so no period mixes pre- and post-rollout observations."""
    d = df.copy()
    d["treated_flag"] = (d.group == "treated").astype(int)
    d["week_bin"] = d["day"] // 7
    pre_cutoff = (rollout_day // 7) * 7
    post_cutoff = pre_cutoff + 7

    pre_weeks = sorted(d.loc[d["day"] < pre_cutoff, "week_bin"].unique())
    post_weeks = sorted(d.loc[d["day"] >= post_cutoff, "week_bin"].unique())
    keep = d["week_bin"].isin(set(pre_weeks) | set(post_weeks))
    panel = (
        d[keep].groupby(["region_id", "week_bin", "treated_flag"], as_index=False)
        .agg(on_time_rate=("on_time_rate", "mean"))
    )
    return panel, pre_weeks, post_weeks


def honest_did_check(df, rollout_day=ROLLOUT_DAY):
    panel, pre_weeks, post_weeks = build_weekly_event_panel(df, rollout_day=rollout_day)
    event_model = dd.MultiPeriodDiD(cluster="region_id", vcov_type="hc1")
    event_results = event_model.fit(
        data=panel, outcome="on_time_rate", treatment="treated_flag", time="week_bin",
        post_periods=post_weeks, unit="region_id", reference_period=pre_weeks[-1],
    )
    hd = dd.HonestDiD(method="relative_magnitude")
    sensitivity = hd.sensitivity_analysis(
        event_results, M_grid=list(np.round(np.arange(0.0, 1.02, 0.02), 2))
    )
    return {
        "avg_att": event_results.avg_att,
        "avg_ci": event_results.avg_conf_int,
        "sensitivity": sensitivity,
        "breakdown_M": sensitivity.breakdown_M,
    }


def sensitivity_chart(sensitivity, source_note):
    df_sens = sensitivity.to_dataframe()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(df_sens["M"], df_sens["ci_lb"] * 100, df_sens["ci_ub"] * 100,
                     color=SLATE, alpha=0.18, label="Robust 95% CI")
    ax.plot(df_sens["M"], df_sens["lb"] * 100, color=SLATE, linewidth=1.6)
    ax.plot(df_sens["M"], df_sens["ub"] * 100, color=SLATE, linewidth=1.6)
    ax.axhline(0, color=GREY, linewidth=1)
    if sensitivity.has_breakdown:
        ax.axvline(sensitivity.breakdown_M, color=MUTED_RED, linestyle="--", linewidth=1.3)
        ax.text(sensitivity.breakdown_M + 0.02, ax.get_ylim()[0], f"breakdown M={sensitivity.breakdown_M:.2f}",
                fontsize=9, color=MUTED_RED, va="bottom")
    style_ax(ax, title="How much parallel-trends violation would it take to explain away the effect?",
             subtitle="Honest DiD (Rambachan-Roth) robust CI vs. the relative-magnitude restriction M",
             xlabel="M (post-period violation, as a multiple of the largest pre-period deviation)",
             ylabel="On-time repayment effect (pp)")
    ax.legend(fontsize=9.5, loc="upper left")
    savefig(fig, FIG_DIR / "did_honest_sensitivity.png", footnote=source_note)


def robustness_checks(df, source_note):
    boot = wild_cluster_bootstrap_check(df)
    print(f"Primary DiD coefficient: {boot['coef']*100:.2f}pp")
    print(f"Analytical cluster-robust 95% CI: [{boot['analytical_ci'][0]*100:.2f}, {boot['analytical_ci'][1]*100:.2f}]pp")
    print(f"Wild cluster bootstrap 95% CI ({boot['n_clusters']} clusters, 999 reps): "
          f"[{boot['bootstrap_ci'][0]*100:.2f}, {boot['bootstrap_ci'][1]*100:.2f}]pp, p={boot['bootstrap_p']:.3f}")

    honest = honest_did_check(df)
    print(f"Event-study average post-period ATT: {honest['avg_att']*100:.2f}pp, "
          f"95% CI [{honest['avg_ci'][0]*100:.2f}, {honest['avg_ci'][1]*100:.2f}]pp")
    print(f"Honest DiD breakdown M: {honest['breakdown_M']:.2f} "
          "(the effect stays distinguishable from zero only while a plausible undetected "
          "pre-trends violation stays under this many times the largest pre-period wobble actually observed)")

    sensitivity_chart(honest["sensitivity"], source_note)
    return {"bootstrap": boot, "honest": honest}


def main():
    df = pd.read_csv(DATA_DIR / "regional_rollout.csv")
    n_regions = df.region_id.nunique()
    source_note = f"Source: synthetic BNPL regional panel · {n_regions} regions, {df.day.nunique()} days"
    robustness_checks(df, source_note)
    print("Wrote reports/figures/did_honest_sensitivity.png")


if __name__ == "__main__":
    main()
