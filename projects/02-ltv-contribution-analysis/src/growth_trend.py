"""Monthly GMV trend with month-over-month growth and a trailing
3-month moving average, via sql/05_kpi_trend_with_deltas.sql (LAG() and
a moving window frame, run through DuckDB)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from db import get_connection, run_sql_file
from style import set_style, style_ax, savefig, SLATE, MUTED_AMBER, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()


def main():
    con = get_connection()
    trend = run_sql_file(con, "05_kpi_trend_with_deltas.sql")
    trend.to_csv(BASE / "reports" / "kpi_trend_with_deltas.csv", index=False)

    n_customers = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    SOURCE = f"Source: synthetic BNPL order data (sql/05_kpi_trend_with_deltas.sql via DuckDB) · n = {n_customers:,} customers"

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(trend.month_index, trend.gmv_usd, color=SLATE, alpha=0.45, width=0.7,
           zorder=2, label="Monthly GMV")
    ax.plot(trend.month_index, trend.gmv_3mo_moving_avg, color=MUTED_AMBER,
             linewidth=2, marker="o", markersize=3.5, zorder=3, label="3-month moving average")
    style_ax(ax, title="Growth is real, but month-over-month gains are decelerating",
             subtitle="Monthly GMV vs. a trailing 3-month moving average, smoothing out MoM noise",
             xlabel="Month", ylabel="GMV (USD)")
    ax.legend(loc="upper left", fontsize=9)
    savefig(fig, FIG_DIR / "gmv_trend_moving_average.png", footnote=SOURCE)

    early = trend[trend.month_index.between(2, 6)]["mom_growth_pct"].mean()
    late = trend[trend.month_index.between(20, 24)]["mom_growth_pct"].mean()
    print(f"Avg MoM growth, months 2-6: {early:.1f}% -> months 20-24: {late:.1f}%")
    print("Wrote reports/kpi_trend_with_deltas.csv and 1 figure.")


if __name__ == "__main__":
    main()
