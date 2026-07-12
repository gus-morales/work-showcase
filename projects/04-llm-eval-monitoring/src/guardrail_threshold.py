"""
Picks the auto-send vs. route-to-human-review threshold on a safety
classifier's risk score, from actual costs rather than a default 0.5 cutoff.
Structurally identical to project 01's cost-based delinquency threshold:
sweep the threshold, price both error types, take the minimum-cost point.

Cost assumptions:
- A bad reply that gets auto-sent (false negative): remediation, trust
  damage, possible escalation. Priced high.
- A fine reply that gets routed to a human anyway (false positive, or any
  reply above the threshold): reviewer time. Priced low, but it's paid on
  every single routed reply, good or bad, so it adds up fast at a low
  threshold.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY, INK

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

COST_BAD_AUTO_SEND = 150.0   # remediation + trust cost if a bad reply reaches the customer unreviewed
COST_HUMAN_REVIEW = 2.0      # reviewer time per reply routed for review, regardless of outcome


def find_cost_optimal_threshold(risk_score, true_bad, thresholds=None):
    """Pure computation. At each threshold: replies at or above it are
    routed to a human (cost COST_HUMAN_REVIEW each); replies below it are
    auto-sent (cost COST_BAD_AUTO_SEND only if truly bad)."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        routed = risk_score >= t
        auto_sent_bad = (~routed) & (true_bad == 1)
        cost = routed.sum() * COST_HUMAN_REVIEW + auto_sent_bad.sum() * COST_BAD_AUTO_SEND
        costs.append(cost)
    costs = np.array(costs)
    best_idx = costs.argmin()
    return thresholds, costs, float(thresholds[best_idx]), float(costs[best_idx])


def guardrail_threshold(df, source_note):
    risk_score = df["risk_score"].values
    true_bad = df["true_bad"].values

    auc = roc_auc_score(true_bad, risk_score)
    thresholds, costs, best_t, best_cost = find_cost_optimal_threshold(risk_score, true_bad)
    default_idx = np.argmin(np.abs(thresholds - 0.50))
    default_cost = costs[default_idx]
    cost_reduction = 1 - best_cost / default_cost

    routed_at_best = (risk_score >= best_t).mean()
    routed_at_default = (risk_score >= 0.50).mean()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, costs, color=SLATE, linewidth=1.8)
    ax.axvline(best_t, color=MUTED_RED, linewidth=1.2, ls="--")
    ax.scatter([best_t], [best_cost], color=MUTED_RED, s=50, zorder=4)
    ax.annotate(f"cost-optimal: t={best_t:.2f}\n${best_cost:,.0f}", xy=(best_t, best_cost),
                xytext=(best_t + 0.08, best_cost + (costs.max() - costs.min()) * 0.08),
                fontsize=9.5, color=MUTED_RED)
    style_ax(ax, title=f"Cost-optimal threshold cuts expected cost {cost_reduction:.0%} vs. a naive 0.50 cutoff",
             subtitle=f"Expected cost per {len(df):,} replies, routed vs. auto-sent",
             xlabel="Route-to-human threshold on risk score", ylabel="Expected total cost (USD)", grid_axis="both")
    savefig(fig, FIG_DIR / "guardrail_cost_curve.png", footnote=source_note)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ["Naive (t=0.50)", f"Cost-optimal (t={best_t:.2f})"]
    vals = [default_cost, best_cost]
    colors = [GREY, SLATE]
    ax.bar(bars, vals, color=colors, width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"${v:,.0f}", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Expected cost, naive vs. cost-optimal threshold",
             subtitle=f"n = {len(df):,} replies", ylabel="Expected total cost (USD)")
    savefig(fig, FIG_DIR / "guardrail_threshold_comparison.png", footnote=source_note)

    print(f"Classifier AUC: {auc:.3f}")
    print(f"Cost-optimal threshold: {best_t:.2f} (${best_cost:,.0f} vs ${default_cost:,.0f} at t=0.50, "
          f"{cost_reduction:.1%} lower)")
    print(f"Share routed to human review: {routed_at_best:.1%} at optimal threshold "
          f"vs. {routed_at_default:.1%} at t=0.50")
    return {
        "auc": float(auc), "best_threshold": best_t, "best_cost": best_cost,
        "default_cost": float(default_cost), "cost_reduction": float(cost_reduction),
        "routed_at_best": float(routed_at_best), "routed_at_default": float(routed_at_default),
    }


def main():
    df = pd.read_csv(DATA_DIR / "guardrail_scores.csv")
    source_note = f"Source: synthetic guardrail scoring data · n = {len(df):,} replies"
    guardrail_threshold(df, source_note)
    print("Wrote reports/figures/guardrail_cost_curve.png, guardrail_threshold_comparison.png")


if __name__ == "__main__":
    main()
