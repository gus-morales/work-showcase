"""
Budget-constrained targeting: the growth team can only afford to reach
a fixed share of the customer base with the next campaign. Compares
three ways to pick who gets it: random targeting, a naive "target the
biggest spenders" rule, and ranking by the calibrated propensity score
from propensity_model.py.

Expected responders under each policy is computed from the model's own
calibrated propensity score, since that's the only response-rate
estimate available for customers who were never actually offered
anything. This step isn't re-validating the model (the gains chart and
held-out AUC already did that): it's a downstream decision analysis
built on a model already shown to rank held-out responders well.

Saves:
    reports/targeting_summary.json
    reports/figures/{targeting_policy_comparison, targeting_segment_mix}.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY, PALETTE, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

BUDGET_FRACTION = 0.20


def expected_responders(df, rank_col, budget_n, ascending=False):
    """Sorts by rank_col, takes the top budget_n rows, sums their
    propensity score. Pure function over a plain dataframe with the two
    needed columns, independently testable against a hand-built case."""
    ranked = df.sort_values(rank_col, ascending=ascending)
    return float(ranked.head(budget_n)["propensity_score"].sum())


def main():
    customers = pd.read_csv(BASE / "data" / "customers.csv")
    scores = pd.read_csv(BASE / "reports" / "propensity_scores.csv")
    df = scores.merge(customers[["customer_id", "monetary_90d"]], on="customer_id", how="left")

    n = len(df)
    budget_n = round(BUDGET_FRACTION * n)
    SOURCE = f"Source: synthetic bank customer data · n = {n:,} customers · budget = top {BUDGET_FRACTION:.0%} ({budget_n:,} customers)"

    random_expected = budget_n * df["propensity_score"].mean()
    topspend_expected = expected_responders(df, "monetary_90d", budget_n)
    model_expected = expected_responders(df, "propensity_score", budget_n)

    lift_vs_random = model_expected / random_expected - 1
    lift_vs_topspend = model_expected / topspend_expected - 1
    print(f"Budget: top {BUDGET_FRACTION:.0%} of customers ({budget_n:,} of {n:,})")
    print(f"Expected responders - random: {random_expected:.0f} | top-spend: {topspend_expected:.0f} | "
          f"model-ranked: {model_expected:.0f}")
    print(f"Model lift vs. random: {lift_vs_random:.1%} | vs. top-spend: {lift_vs_topspend:.1%}")

    # --- Policy comparison chart ---
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    policies = ["Random\ntargeting", "Top-spend\ntargeting", "Model-ranked\ntargeting"]
    vals = [random_expected, topspend_expected, model_expected]
    ax.bar(policies, vals, color=[GREY, MUTED_RED, SLATE], width=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.015, f"{v:.0f}", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Ranking by propensity beats spend-based targeting under the same budget",
             subtitle=f"Expected responders captured, top {BUDGET_FRACTION:.0%} of customers targeted",
             ylabel="Expected responders")
    savefig(fig, FIG_DIR / "targeting_policy_comparison.png", footnote=SOURCE)

    # --- Segment mix: who does each policy actually target? ---
    overall_mix = df["segment"].value_counts(normalize=True) * 100
    model_top = df.sort_values("propensity_score", ascending=False).head(budget_n)
    topspend_top = df.sort_values("monetary_90d", ascending=False).head(budget_n)
    model_mix = model_top["segment"].value_counts(normalize=True) * 100
    topspend_mix = topspend_top["segment"].value_counts(normalize=True) * 100

    segments = list(overall_mix.index)
    mix_df = pd.DataFrame({
        "Overall population": overall_mix.reindex(segments).fillna(0),
        "Top-spend targeting": topspend_mix.reindex(segments).fillna(0),
        "Model-ranked targeting": model_mix.reindex(segments).fillna(0),
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(segments))
    width = 0.26
    for i, col in enumerate(mix_df.columns):
        ax.bar(x + (i - 1) * width, mix_df[col], width, color=PALETTE[i], label=col, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=15, ha="right", fontsize=9)
    style_ax(ax, title="The model shifts budget toward Lapsed and Occasional segments",
             subtitle="Segment mix of the targeted group under each policy, vs. the overall population",
             ylabel="Share of targeted group (%)")
    ax.legend(loc="upper right", fontsize=9)
    savefig(fig, FIG_DIR / "targeting_segment_mix.png", footnote=SOURCE)

    summary = {
        "n_customers": n, "budget_fraction": BUDGET_FRACTION, "budget_n": int(budget_n),
        "expected_responders_random": round(random_expected, 1),
        "expected_responders_topspend": round(topspend_expected, 1),
        "expected_responders_model": round(model_expected, 1),
        "lift_vs_random": round(lift_vs_random, 4),
        "lift_vs_topspend": round(lift_vs_topspend, 4),
        "model_segment_mix_pct": model_mix.round(1).to_dict(),
        "topspend_segment_mix_pct": topspend_mix.round(1).to_dict(),
        "overall_segment_mix_pct": overall_mix.round(1).to_dict(),
    }
    with open(BASE / "reports" / "targeting_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
