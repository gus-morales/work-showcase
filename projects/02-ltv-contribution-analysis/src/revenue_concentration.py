"""Revenue concentration within each acquisition cohort: what share of
a cohort's total revenue comes from its top 5% of customers by spend,
via sql/06_top_customers_by_cohort.sql (DENSE_RANK partitioned by
cohort, run through DuckDB)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from db import get_connection, run_sql_file
from style import set_style, style_ax, savefig, SLATE, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()


def main():
    con = get_connection()
    conc = run_sql_file(con, "06_top_customers_by_cohort.sql")
    conc.to_csv(BASE / "reports" / "revenue_concentration_by_cohort.csv", index=False)

    n_customers = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    SOURCE = f"Source: synthetic BNPL customer data (sql/06_top_customers_by_cohort.sql via DuckDB) · n = {n_customers:,} customers"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(conc.cohort_month, conc.top5pct_revenue_share * 100, color=SLATE, width=0.7, zorder=3)
    mean_share = conc.top5pct_revenue_share.mean() * 100
    ax.axhline(mean_share, color=GREY, linestyle="--", linewidth=1.1,
               label=f"Average ({mean_share:.0f}%)")
    style_ax(ax, title="A cohort's top 5% of customers drive about a quarter of its revenue",
             subtitle="Share of cohort revenue from its top 5% of customers by spend, DENSE_RANK per cohort",
             xlabel="Acquisition cohort (month)", ylabel="Share of cohort revenue (%)")
    ax.legend(loc="upper right", fontsize=9)
    savefig(fig, FIG_DIR / "revenue_concentration_by_cohort.png", footnote=SOURCE)

    print(f"Average top-5% revenue share across cohorts: {mean_share:.1f}%")
    print(f"Range: {conc.top5pct_revenue_share.min()*100:.1f}% to {conc.top5pct_revenue_share.max()*100:.1f}%")
    print("Wrote reports/revenue_concentration_by_cohort.csv and 1 figure.")


if __name__ == "__main__":
    main()
