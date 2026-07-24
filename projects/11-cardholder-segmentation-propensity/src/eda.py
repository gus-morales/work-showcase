"""Exploratory analysis on the raw customer data, before any clustering or
model gets trained: saves charts to reports/figures and a numeric summary
to reports/eda_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()


def main():
    df = pd.read_csv(BASE / "data" / "customers.csv")
    offered = df[df["past_offer_sent"] == 1].copy()
    SOURCE_OFFERED = f"Source: synthetic bank customer data · n = {len(offered):,} customers offered the past campaign"

    # 1. Response rate by recency band: the "win-back sweet spot". A
    # moderately lapsed customer responds better than either a
    # currently-active one (nothing to win back) or a fully dormant one
    # (too far gone), which is the whole reason a flat "target every
    # lapsed customer" rule undersells what a propensity model can do.
    offered["recency_band"] = pd.cut(
        offered["recency_days"], [-1, 29, 120, 181], labels=["Active (<30d)", "Lapsed (30-120d)", "Dormant (>120d)"],
    )
    band_rates = offered.groupby("recency_band", observed=True)["responded"].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = [GREY, MUTED_RED, SLATE]
    ax.bar(band_rates.index.astype(str), band_rates.values, color=colors, width=0.55, zorder=3)
    for i, v in enumerate(band_rates.values):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Moderately lapsed customers respond best, not the most active or the most dormant",
             subtitle="Response rate to the past campaign, by recency band",
             ylabel="Response rate (%)")
    savefig(fig, FIG_DIR / "response_rate_by_recency_band.png", footnote=SOURCE_OFFERED)

    # 2. Response rate by lifetime-orders tercile, a proxy for long-run
    # customer value that current-window behavior alone doesn't carry.
    offered["value_tercile"] = pd.qcut(offered["lifetime_orders"], 3, labels=["Low", "Mid", "High"])
    value_rates = offered.groupby("value_tercile", observed=True)["responded"].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(value_rates.index.astype(str), value_rates.values, color=SLATE, width=0.55, zorder=3)
    for i, v in enumerate(value_rates.values):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Response climbs with lifetime order count, a proxy for long-run value",
             subtitle="Response rate by lifetime-orders tercile",
             xlabel="Lifetime orders tercile", ylabel="Response rate (%)")
    savefig(fig, FIG_DIR / "response_rate_by_value_tercile.png", footnote=SOURCE_OFFERED)

    # 3. Decline rate carries no real response signal (it's a decoy
    # feature by construction): shown here as a flat line across
    # terciles, worth checking before it gets fed into a model at all.
    offered["decline_tercile"] = pd.qcut(offered["decline_rate"], 3, labels=["Low", "Mid", "High"])
    decline_rates = offered.groupby("decline_tercile", observed=True)["responded"].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(decline_rates.index.astype(str), decline_rates.values, color=GREY, width=0.55, zorder=3)
    for i, v in enumerate(decline_rates.values):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(decline_rates.values) * 1.4)
    style_ax(ax, title="Authorization decline rate carries no real response signal",
             subtitle="Response rate by decline-rate tercile, roughly flat across all three",
             xlabel="Decline rate tercile", ylabel="Response rate (%)")
    savefig(fig, FIG_DIR / "response_rate_by_decline_tercile.png", footnote=SOURCE_OFFERED)

    # Summary markdown
    lines = ["# EDA summary\n"]
    lines.append(f"- Customers: {len(df):,}\n")
    lines.append(f"- Past offer sent: {len(offered):,} ({len(offered) / len(df):.1%} of customers)\n")
    lines.append(f"- Response rate among offered: {offered['responded'].mean():.1%}\n")
    lines.append(f"- Response rate, lapsed vs. active vs. dormant: "
                 f"{band_rates.get('Lapsed (30-120d)', 0):.1f}% vs. "
                 f"{band_rates.get('Active (<30d)', 0):.1f}% vs. "
                 f"{band_rates.get('Dormant (>120d)', 0):.1f}%\n")
    lines.append(f"- Response rate, high vs. low lifetime-orders tercile: "
                 f"{value_rates.get('High', 0):.1f}% vs. {value_rates.get('Low', 0):.1f}%\n")
    lines.append(f"- Response rate, high vs. low decline-rate tercile (decoy feature): "
                 f"{decline_rates.get('High', 0):.1f}% vs. {decline_rates.get('Low', 0):.1f}%\n")
    (BASE / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("EDA complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
