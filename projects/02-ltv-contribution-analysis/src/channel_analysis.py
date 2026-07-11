"""Acquisition channel quality and mix-shift analysis, via
sql/03_channel_quality.sql and sql/04_channel_mix_shift.sql (DuckDB)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db import get_connection, run_sql_file
from style import set_style, style_ax, savefig, PALETTE, SLATE, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

CHANNEL_LABELS = {
    "partner_store": "Partner store", "referral": "Referral",
    "organic": "Organic", "paid_social": "Paid social",
}
CHANNEL_ORDER = ["partner_store", "referral", "organic", "paid_social"]


def main():
    con = get_connection()
    quality = run_sql_file(con, "03_channel_quality.sql")
    mix = run_sql_file(con, "04_channel_mix_shift.sql")
    quality.to_csv(BASE / "reports" / "channel_quality.csv", index=False)
    mix.to_csv(BASE / "reports" / "channel_mix_shift.csv", index=False)

    n_customers = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    SOURCE = f"Source: synthetic BNPL customer data (sql/03-04, via DuckDB) · n = {n_customers:,} customers"

    # --- Channel quality bar chart ---
    q = quality.set_index("acquisition_channel").reindex(
        quality.sort_values("avg_revenue_per_customer_usd", ascending=False)["acquisition_channel"])
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [CHANNEL_LABELS[c] for c in q.index]
    ax.bar(labels, q["avg_revenue_per_customer_usd"], color=PALETTE[:len(q)], width=0.55, zorder=3)
    for i, v in enumerate(q["avg_revenue_per_customer_usd"]):
        ax.text(i, v + 2, f"${v:,.0f}", ha="center", fontsize=10, color="#333")
    style_ax(ax, title="Partner-store customers are worth 2-3x paid-social customers",
             subtitle="Average lifetime revenue per acquired customer, to date",
             ylabel="Avg revenue per customer (USD)")
    savefig(fig, FIG_DIR / "channel_quality.png", footnote=SOURCE)

    # --- Channel mix shift over cohorts (stacked area) ---
    pivot = mix.pivot(index="cohort_month", columns="acquisition_channel", values="channel_share").fillna(0)
    pivot = pivot[[c for c in CHANNEL_ORDER if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.stackplot(pivot.index, [pivot[c] for c in pivot.columns],
                 labels=[CHANNEL_LABELS[c] for c in pivot.columns],
                 colors=PALETTE[:len(pivot.columns)], alpha=0.9)
    style_ax(ax, title="Paid social's share of new cohorts has roughly tripled",
             subtitle="Acquisition channel mix by cohort month",
             xlabel="Acquisition cohort (month)", ylabel="Share of cohort")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    savefig(fig, FIG_DIR / "channel_mix_shift.png", footnote=SOURCE)

    print(quality.to_string(index=False))
    print(f"\nWrote reports/channel_quality.csv, channel_mix_shift.csv, and 2 figures.")


if __name__ == "__main__":
    main()
