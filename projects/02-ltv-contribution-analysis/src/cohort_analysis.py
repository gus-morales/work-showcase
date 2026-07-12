"""Cohort retention and revenue analysis, via sql/01_cohort_revenue.sql
run through DuckDB directly against the generated CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from db import get_connection, run_sql_file
from style import set_style, style_ax, savefig, SLATE, HEATMAP_CMAP, HEATMAP_TEXT_LOW, HEATMAP_TEXT_HIGH

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

MAX_MONTHS_SINCE = 11  # cap the retention matrix so every cohort shown has full data


def main():
    con = get_connection()
    cohort = run_sql_file(con, "01_cohort_revenue.sql")
    cohort.to_csv(BASE / "reports" / "cohort_revenue.csv", index=False)

    n_customers = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    SOURCE = f"Source: synthetic BNPL order data (sql/01_cohort_revenue.sql via DuckDB) · n = {n_customers:,} customers"

    # --- Retention heatmap: only cohorts with >= MAX_MONTHS_SINCE+1 months observed ---
    cohort_ages = cohort.groupby("cohort_month")["months_since_acquisition"].max()
    full_cohorts = cohort_ages[cohort_ages >= MAX_MONTHS_SINCE].index
    mat = cohort[cohort.cohort_month.isin(full_cohorts) & (cohort.months_since_acquisition <= MAX_MONTHS_SINCE)]
    pivot = mat.pivot(index="cohort_month", columns="months_since_acquisition", values="retention_rate")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pivot.values, cmap=HEATMAP_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                         color=HEATMAP_TEXT_HIGH if v > 0.5 else HEATMAP_TEXT_LOW)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    style_ax(ax, title="Cohort retention decays fastest in month 1",
             subtitle="Share of originally-acquired customers still ordering, by cohort",
             xlabel="Months since acquisition", ylabel="Acquisition cohort (month)", grid_axis=None)
    savefig(fig, FIG_DIR / "cohort_retention_heatmap.png", footnote=SOURCE)

    # --- Revenue-per-acquired-customer curve, averaged across full cohorts ---
    avg_curve = mat.groupby("months_since_acquisition")["revenue_per_acquired_customer"].mean()
    cum_curve = avg_curve.cumsum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cum_curve.index, cum_curve.values, color=SLATE, linewidth=1.8, marker="o", markersize=4.5)
    style_ax(ax, title="Cumulative revenue per acquired customer",
             subtitle=f"Averaged across cohorts with {MAX_MONTHS_SINCE + 1}+ months of observed history",
             xlabel="Months since acquisition", ylabel="Cumulative revenue (USD)")
    savefig(fig, FIG_DIR / "cohort_cumulative_revenue.png", footnote=SOURCE)

    print(f"Full cohorts used (>= {MAX_MONTHS_SINCE + 1} months observed): {sorted(full_cohorts.tolist())}")
    print(f"Month-1 retention (avg across full cohorts): "
          f"{mat[mat.months_since_acquisition == 1]['retention_rate'].mean():.1%}")
    print(f"Month-{MAX_MONTHS_SINCE} cumulative revenue per acquired customer: {cum_curve.iloc[-1]:.2f} USD")
    print("Wrote reports/cohort_revenue.csv and 2 figures.")


if __name__ == "__main__":
    main()
