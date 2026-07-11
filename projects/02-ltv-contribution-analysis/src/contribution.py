"""
Contribution (driver) analysis: decomposes the change in GMV between two
periods into three drivers - active customers, orders per customer, and
average order value - using a log-share decomposition (each driver's
log-growth share of total log-growth is allocated that share of the
dollar change). Standard approach for a multiplicative KPI tree
(GMV = customers x orders/customer x avg order value).

Monthly KPI aggregation comes from sql/02_monthly_kpis.sql via DuckDB;
the decomposition arithmetic is plain pandas/numpy.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from db import get_connection, run_sql_file
from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_AMBER, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()


def decompose(row_a, row_b):
    """Log-share decomposition of the GMV change between two monthly KPI rows."""
    log_c = np.log(row_b["active_customers"] / row_a["active_customers"])
    log_f = np.log(row_b["orders_per_customer"] / row_a["orders_per_customer"])
    log_v = np.log(row_b["avg_order_value_usd"] / row_a["avg_order_value_usd"])
    total_log = log_c + log_f + log_v
    delta_gmv = row_b["gmv_usd"] - row_a["gmv_usd"]

    if abs(total_log) < 1e-9:
        shares = {"customers": 1 / 3, "frequency": 1 / 3, "avg_order_value": 1 / 3}
    else:
        shares = {"customers": log_c / total_log, "frequency": log_f / total_log,
                  "avg_order_value": log_v / total_log}
    return {k: v * delta_gmv for k, v in shares.items()}, delta_gmv


def main():
    con = get_connection()
    kpis = run_sql_file(con, "02_monthly_kpis.sql")
    kpis.to_csv(BASE / "reports" / "monthly_kpis.csv", index=False)

    n_customers = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    SOURCE = f"Source: synthetic BNPL order data (sql/02_monthly_kpis.sql via DuckDB) · n = {n_customers:,} customers"

    # --- Headline waterfall: month 4 (early, stable) vs month 20 (late, mix-shifted) ---
    row_a = kpis[kpis.month_index == 4].iloc[0]
    row_b = kpis[kpis.month_index == 20].iloc[0]
    contributions, delta_gmv = decompose(row_a, row_b)

    labels = ["Month 4\nGMV", "Active\ncustomers", "Orders per\ncustomer", "Avg order\nvalue", "Month 20\nGMV"]
    values = [row_a["gmv_usd"], contributions["customers"], contributions["frequency"],
              contributions["avg_order_value"], row_b["gmv_usd"]]

    cumulative = [values[0]]
    for v in values[1:-1]:
        cumulative.append(cumulative[-1] + v)
    cumulative.append(values[-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_bottoms = [0, cumulative[0], cumulative[1], cumulative[2], 0]
    bar_heights = [values[0], values[1], values[2], values[3], values[4]]
    bottoms_plot = [0, min(cumulative[0], cumulative[1]), min(cumulative[1], cumulative[2]),
                    min(cumulative[2], cumulative[3]), 0]
    heights_plot = [values[0], abs(values[1]), abs(values[2]), abs(values[3]), values[4]]
    colors = [SLATE, (MUTED_TEAL if values[1] >= 0 else MUTED_RED),
              (MUTED_TEAL if values[2] >= 0 else MUTED_RED),
              (MUTED_TEAL if values[3] >= 0 else MUTED_RED), SLATE]

    ax.bar(range(5), heights_plot, bottom=bottoms_plot, color=colors, width=0.6, zorder=3)
    for i, (v, b) in enumerate(zip(values, bottoms_plot)):
        label_y = b + heights_plot[i] + (max(values[0], values[-1]) * 0.02)
        sign = "+" if (0 < i < 4 and v >= 0) else ("" if i in (0, 4) else "-")
        ax.text(i, label_y, f"{sign}{abs(v):,.0f}", ha="center", fontsize=10, color="#333")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    style_ax(ax, title="What drove GMV from month 4 to month 20",
             subtitle="Log-share decomposition into customers, order frequency, and order value",
             ylabel="GMV (USD)")
    savefig(fig, FIG_DIR / "contribution_waterfall.png", footnote=SOURCE)

    # --- Monthly driver contributions over time (stacked) ---
    monthly_contribs = []
    for m in range(2, int(kpis.month_index.max()) + 1):
        prev = kpis[kpis.month_index == m - 1]
        curr = kpis[kpis.month_index == m]
        if prev.empty or curr.empty:
            continue
        c, delta = decompose(prev.iloc[0], curr.iloc[0])
        c["month_index"] = m
        c["delta_gmv"] = delta
        monthly_contribs.append(c)
    mc = pd.DataFrame(monthly_contribs)
    mc.to_csv(BASE / "reports" / "monthly_contributions.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(mc.month_index, mc.customers, color=SLATE, label="Active customers", width=0.7, zorder=3)
    ax.bar(mc.month_index, mc.frequency, bottom=mc.customers, color=MUTED_TEAL,
           label="Orders per customer", width=0.7, zorder=3)
    below = mc.customers + mc.frequency
    ax.bar(mc.month_index, mc.avg_order_value, bottom=np.where(mc.avg_order_value >= 0, below, below),
           color=MUTED_AMBER, label="Avg order value", width=0.7, zorder=3)
    ax.plot(mc.month_index, mc.delta_gmv, color="#222222", linewidth=1.6, marker="o", markersize=3.5,
            label="Net GMV change")
    ax.axhline(0, color=GREY, linewidth=1)
    style_ax(ax, title="GMV growth comes almost entirely from new customers, not engagement",
             subtitle="Month-over-month GMV change decomposed by driver",
             xlabel="Month", ylabel="Contribution to MoM GMV change (USD)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    savefig(fig, FIG_DIR / "contribution_monthly.png", footnote=SOURCE)

    print(f"Month 4 GMV: {row_a['gmv_usd']:,.0f} -> Month 20 GMV: {row_b['gmv_usd']:,.0f} "
          f"(delta {delta_gmv:,.0f})")
    print("Contribution breakdown:", {k: round(v, 0) for k, v in contributions.items()})
    print("Wrote reports/monthly_kpis.csv, monthly_contributions.csv, and 2 figures.")


if __name__ == "__main__":
    main()
