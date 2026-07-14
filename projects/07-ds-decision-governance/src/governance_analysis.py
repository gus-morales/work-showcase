"""
How impact level relates to approval speed and to what happens after a
decision ships: two descriptive charts, approval lag and rollback rate,
both by impact level.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

IMPACT_ORDER = ["low", "medium", "high"]


def approval_lag_chart(df, source_note):
    lag = df.groupby("impact_level", observed=True)["approval_lag_days"].mean().reindex(IMPACT_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(lag.index, lag.values, color=SLATE, width=0.5, zorder=3)
    for i, v in enumerate(lag.values):
        ax.text(i, v + lag.max() * 0.02, f"{v:.1f}d", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Higher-impact decisions take longer to approve",
             subtitle="Mean approval lag (proposed to approved) by impact level",
             xlabel="Impact level", ylabel="Approval lag (days)")
    savefig(fig, FIG_DIR / "approval_lag_by_impact.png", footnote=source_note)
    return lag


def rollback_rate_chart(df, source_note):
    resolved = df[df["status"].isin(["closed", "reverted"])]
    rate = resolved.groupby("impact_level", observed=True)["outcome"].apply(
        lambda s: (s == "rollback").mean()
    ).reindex(IMPACT_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(rate.index, rate.values * 100, color=MUTED_RED, width=0.5, zorder=3)
    for i, v in enumerate(rate.values):
        ax.text(i, v * 100 + 0.4, f"{v:.1%}", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Rollback rate falls as impact level rises",
             subtitle="Share of resolved decisions later rolled back, by impact level",
             xlabel="Impact level", ylabel="Rollback rate (%)")
    savefig(fig, FIG_DIR / "rollback_rate_by_impact.png", footnote=source_note)
    return rate


def main():
    df = pd.read_csv(BASE / "data" / "decision_log.csv")
    df["impact_level"] = pd.Categorical(df["impact_level"], categories=IMPACT_ORDER, ordered=True)
    source_note = f"Source: synthetic decision log (src/generate_data.py) · n = {len(df):,} decisions"

    lag = approval_lag_chart(df, source_note)
    rate = rollback_rate_chart(df, source_note)

    print(f"Mean approval lag by impact level:\n{lag.round(2)}")
    print(f"Rollback rate by impact level:\n{rate.round(3)}")
    print("Wrote reports/figures/approval_lag_by_impact.png, rollback_rate_by_impact.png")

    metrics = {
        "n_decisions": int(len(df)),
        "approval_lag_days_mean_by_impact": lag.round(2).to_dict(),
        "rollback_rate_by_impact": rate.round(4).to_dict(),
    }
    (BASE / "reports" / "governance_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
