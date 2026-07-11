"""
Production monitoring for the reply-drafting feature: a p-chart (control
chart for a proportion) on the daily judge-scored acceptable rate. Input
features (ticket volume, category mix) don't shift here; only output
quality does, so this is a case standard input-drift monitoring (PSI on
features, as in project 01) would miss entirely. Catching it requires
scoring outputs directly, the same lesson as project 01's monitoring
section applied to a generative feature instead of a classifier.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

REFERENCE_END_DAY = 59  # days 0-59 establish the "in control" baseline
MIN_CONSECUTIVE_ALERTS = 3  # require a run, not a single noisy day, before flagging


def compute_control_chart(df: pd.DataFrame, reference_end_day: int = REFERENCE_END_DAY) -> dict:
    """Pure computation: standard p-chart with per-day control limits
    (subgroup size varies day to day), plus a simple run rule to avoid
    flagging single-day noise. No plotting, no I/O."""
    df = df.sort_values("day").reset_index(drop=True)
    reference = df[df["day"] <= reference_end_day]

    center = float((reference["n_acceptable"].sum()) / (reference["n_tickets"].sum()))
    se = np.sqrt(center * (1 - center) / df["n_tickets"].values)
    ucl = np.clip(center + 3 * se, 0, 1)
    lcl = np.clip(center - 3 * se, 0, 1)
    out_of_control = df["acceptable_rate"].values < lcl

    # First day that starts a run of >= MIN_CONSECUTIVE_ALERTS consecutive
    # out-of-control days.
    detected_day = None
    run_len = 0
    for i, flagged in enumerate(out_of_control):
        run_len = run_len + 1 if flagged else 0
        if run_len >= MIN_CONSECUTIVE_ALERTS:
            detected_day = int(df["day"].iloc[i - MIN_CONSECUTIVE_ALERTS + 1])
            break

    return {
        "center": center,
        "ucl": ucl,
        "lcl": lcl,
        "out_of_control": out_of_control,
        "detected_day": detected_day,
        "days": df["day"].values,
        "rates": df["acceptable_rate"].values,
    }


def drift_monitoring(df, source_note):
    result = compute_control_chart(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result["days"], result["rates"] * 100, color=SLATE, linewidth=1.4, zorder=3, label="Daily acceptable rate")
    ax.axhline(result["center"] * 100, color=GREY, linewidth=1.2, ls="--", label="Center line (reference period)")
    ax.plot(result["days"], result["ucl"] * 100, color=GREY, linewidth=0.8, ls=":")
    ax.plot(result["days"], result["lcl"] * 100, color=GREY, linewidth=0.8, ls=":", label="Control limits (3-sigma)")
    alert_days = result["days"][result["out_of_control"]]
    alert_rates = result["rates"][result["out_of_control"]] * 100
    ax.scatter(alert_days, alert_rates, color=MUTED_RED, s=22, zorder=4, label="Below control limit")
    if result["detected_day"] is not None:
        ax.axvline(result["detected_day"], color=MUTED_RED, linewidth=1, ls="-", alpha=0.5)
        ax.annotate(f"regression flagged\nday {result['detected_day']}",
                    xy=(result["detected_day"], result["center"] * 100),
                    xytext=(result["detected_day"] + 3, result["center"] * 100 - 18),
                    fontsize=9.5, color=MUTED_RED)
    style_ax(ax, title="A p-chart catches the quality regression within days, not weeks",
             subtitle=f"Daily judge-scored acceptable rate vs. control limits from days 0-{REFERENCE_END_DAY}",
             xlabel="Day", ylabel="Acceptable rate (%)")
    ax.legend(fontsize=9, loc="lower left")
    savefig(fig, FIG_DIR / "quality_control_chart.png", footnote=source_note)

    print(f"Reference-period center line: {result['center']:.1%}")
    print(f"Days below control limit: {int(result['out_of_control'].sum())} / {len(df)}")
    if result["detected_day"] is not None:
        print(f"Regression detected starting day {result['detected_day']} "
              f"({MIN_CONSECUTIVE_ALERTS}+ consecutive out-of-control days)")
    else:
        print("No sustained regression detected.")
    return result


def main():
    df = pd.read_csv(DATA_DIR / "quality_monitoring.csv")
    source_note = f"Source: synthetic monitoring data · {len(df)} days, {df['n_tickets'].sum():,} tickets"
    drift_monitoring(df, source_note)
    print("Wrote reports/figures/quality_control_chart.png")


if __name__ == "__main__":
    main()
