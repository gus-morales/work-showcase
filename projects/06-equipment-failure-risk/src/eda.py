"""Exploratory analysis on the raw truck-telemetry data, before any model
gets trained: saves charts to reports/figures and a numeric summary to
reports/eda_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()


def main():
    tx = pd.read_csv(BASE / "data" / "truck_telemetry.csv", parse_dates=["date"])
    SOURCE = f"Source: synthetic mining fleet telemetry (src/generate_data.py) · n = {len(tx):,} truck-days"

    n_healthy = int((tx["failure_within_7d"] == 0).sum())
    n_failure = int((tx["failure_within_7d"] == 1).sum())
    failure_rate = tx["failure_within_7d"].mean()

    # 1. Class imbalance, on a linear scale on purpose: the failure bar is
    # supposed to look almost invisible next to the healthy one, since
    # that's the actual shape of the problem every metric choice and
    # threshold decision later has to work around.
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ["No failure", "Failure"]
    vals = [n_healthy, n_failure]
    ax.bar(bars, vals, color=[SLATE, MUTED_RED], width=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + len(tx) * 0.01, f"{v:,} ({v / len(tx):.1%})", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title=f"Unplanned failure is {failure_rate:.1%} of truck-days, healthy dwarfs it on a linear scale",
             subtitle="Truck-day count by outcome",
             ylabel="Truck-days")
    savefig(fig, FIG_DIR / "class_imbalance.png", footnote=SOURCE)

    # 2. Failure rate by overdue-maintenance bucket
    tx["pm_bucket"] = pd.cut(tx["days_since_last_pm"], [-1, 14, 29, 44, 59, 1000],
                              labels=["0-14", "15-29", "30-44", "45-59", "60+"])
    pm_rates = tx.groupby("pm_bucket", observed=True)["failure_within_7d"].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(pm_rates.index.astype(str), pm_rates.values, color=SLATE, width=0.55, zorder=3)
    for i, v in enumerate(pm_rates.values):
        ax.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Failure rate climbs sharply once maintenance is overdue",
             subtitle="Failure rate by days since last preventive-maintenance service",
             xlabel="Days since last PM", ylabel="Failure rate (%)")
    savefig(fig, FIG_DIR / "failure_rate_by_pm_overdue.png", footnote=SOURCE)

    # 3. Failure rate by recent fault codes
    tx["fault_bucket"] = pd.cut(tx["fault_codes_7d"], [-1, 0, 1, 100],
                                 labels=["0", "1", "2+"])
    fault_rates = tx.groupby("fault_bucket", observed=True)["failure_within_7d"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(fault_rates.index.astype(str), fault_rates.values, color=SLATE, width=0.5, zorder=3)
    for i, v in enumerate(fault_rates.values):
        ax.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Recent fault codes are one of the strongest single failure signals",
             subtitle="Failure rate by diagnostic trouble codes in the last 7 days",
             xlabel="Fault codes in the last 7 days", ylabel="Failure rate (%)")
    savefig(fig, FIG_DIR / "failure_rate_by_fault_codes.png", footnote=SOURCE)

    # Summary markdown
    lines = ["# EDA summary\n"]
    lines.append(f"- Truck-days: {len(tx):,}, Trucks: {tx['truck_id'].nunique():,}\n")
    lines.append(f"- Failure rate: {failure_rate:.3%} ({n_failure:,} failure vs. {n_healthy:,} healthy, "
                 f"a {n_healthy / n_failure:.0f}:1 ratio)\n")
    lines.append(f"- Failure rate, 60+ days since last PM vs. 0-14: "
                 f"{pm_rates.loc['60+']:.2f}% vs. {pm_rates.loc['0-14']:.2f}%\n")
    lines.append(f"- Failure rate, 2+ fault codes in the last 7 days vs. 0: "
                 f"{fault_rates.loc['2+']:.2f}% vs. {fault_rates.loc['0']:.2f}%\n")
    (BASE / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("EDA complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
