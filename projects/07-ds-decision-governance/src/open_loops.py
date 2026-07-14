"""
Monthly control chart on the metric-check on-time rate: of the decisions
whose 30-day metric check came due in a given month, what share got
closed by its due date? Same p-chart approach as project 04's quality
control chart, applied to a monitoring-follow-through rate instead of a
judge-scored acceptable rate.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

REFERENCE_MONTH_END = 8  # months 0-8 establish the "in control" baseline
MIN_CONSECUTIVE_ALERTS = 3  # require a run, not a single noisy month, before flagging
MIN_MONTHLY_N = 15  # drop months with too few checks due for a stable rate


def monthly_on_time_rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month a metric check came due, with the count due and
    the count closed on time that month. The trailing month is almost
    always partial (the data window ends mid-month), so it's dropped
    along with any other month too thin to give a stable rate."""
    due = df[df["metric_check_due"].notna()].copy()
    due["metric_check_due"] = pd.to_datetime(due["metric_check_due"])
    due["metric_check_on_time"] = due["metric_check_on_time"].astype("boolean")
    # Month index: whole calendar months elapsed since the earliest due date.
    start = due["metric_check_due"].min().to_period("M")
    due["month"] = due["metric_check_due"].dt.to_period("M").apply(lambda p: (p - start).n)

    monthly = due.groupby("month").agg(
        n_due=("metric_check_on_time", "size"),
        n_on_time=("metric_check_on_time", "sum"),
    ).reset_index()
    monthly["on_time_rate"] = monthly["n_on_time"] / monthly["n_due"]
    return monthly[monthly["n_due"] >= MIN_MONTHLY_N].reset_index(drop=True)


def compute_control_chart(monthly: pd.DataFrame, reference_month_end: int = REFERENCE_MONTH_END) -> dict:
    """Pure computation: standard p-chart with per-month control limits
    (subgroup size varies month to month), plus a simple run rule to
    avoid flagging single-month noise. No plotting, no I/O."""
    monthly = monthly.sort_values("month").reset_index(drop=True)
    reference = monthly[monthly["month"] <= reference_month_end]

    center = float(reference["n_on_time"].sum() / reference["n_due"].sum())
    se = np.sqrt(center * (1 - center) / monthly["n_due"].values)
    ucl = np.clip(center + 3 * se, 0, 1)
    lcl = np.clip(center - 3 * se, 0, 1)
    out_of_control = monthly["on_time_rate"].values < lcl

    detected_month = None
    run_len = 0
    for i, flagged in enumerate(out_of_control):
        run_len = run_len + 1 if flagged else 0
        if run_len >= MIN_CONSECUTIVE_ALERTS:
            detected_month = int(monthly["month"].iloc[i - MIN_CONSECUTIVE_ALERTS + 1])
            break

    return {
        "center": center,
        "ucl": ucl,
        "lcl": lcl,
        "out_of_control": out_of_control,
        "detected_month": detected_month,
        "months": monthly["month"].values,
        "rates": monthly["on_time_rate"].values,
    }


def open_loops_chart(df, source_note):
    monthly = monthly_on_time_rates(df)
    result = compute_control_chart(monthly)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result["months"], result["rates"] * 100, color=SLATE, linewidth=1.4, zorder=3, label="Monthly on-time rate")
    ax.axhline(result["center"] * 100, color=GREY, linewidth=1.2, ls="--", label="Center line (reference period)")
    ax.plot(result["months"], result["ucl"] * 100, color=GREY, linewidth=0.8, ls=":")
    ax.plot(result["months"], result["lcl"] * 100, color=GREY, linewidth=0.8, ls=":", label="Control limits (3-sigma)")
    alert_months = result["months"][result["out_of_control"]]
    alert_rates = result["rates"][result["out_of_control"]] * 100
    ax.scatter(alert_months, alert_rates, color=MUTED_RED, s=22, zorder=4, label="Below control limit")
    if result["detected_month"] is not None:
        ax.axvline(result["detected_month"], color=MUTED_RED, linewidth=1, ls="-", alpha=0.5)
        ax.annotate(f"backlog flagged\nmonth {result['detected_month']}",
                    xy=(result["detected_month"], result["center"] * 100),
                    xytext=(result["detected_month"] + 1.5, result["center"] * 100 + 8),
                    fontsize=9.5, color=MUTED_RED)
    style_ax(ax, title="A control chart catches the follow-up backlog within a few months",
             subtitle=f"Monthly metric-check on-time rate vs. control limits from months 0-{REFERENCE_MONTH_END}",
             xlabel="Month", ylabel="On-time rate (%)")
    ax.legend(fontsize=9, loc="lower left")
    savefig(fig, FIG_DIR / "monitoring_control_chart.png", footnote=source_note)

    return result, monthly


def main():
    df = pd.read_csv(BASE / "data" / "decision_log.csv")
    source_note = f"Source: synthetic decision log (src/generate_data.py) · n = {df['metric_check_due'].notna().sum():,} metric checks due"
    result, monthly = open_loops_chart(df, source_note)

    print(f"Reference-period center line: {result['center']:.1%}")
    print(f"Months below control limit: {int(result['out_of_control'].sum())} / {len(monthly)}")
    if result["detected_month"] is not None:
        print(f"Backlog flagged starting month {result['detected_month']} "
              f"({MIN_CONSECUTIVE_ALERTS}+ consecutive out-of-control months)")
    else:
        print("No sustained backlog detected.")
    print("Wrote reports/figures/monitoring_control_chart.png")

    on_time = df["metric_check_on_time"].astype("boolean")
    on_time_by_impact = (
        on_time[on_time.notna()].groupby(df.loc[on_time.notna(), "impact_level"]).mean()
        .reindex(["low", "medium", "high"])
    )
    print(f"Metric-check on-time rate by impact level:\n{on_time_by_impact.round(3)}")

    metrics = {
        "reference_center": round(result["center"], 4),
        "months_below_control_limit": int(result["out_of_control"].sum()),
        "months_total": int(len(monthly)),
        "backlog_detected_month": result["detected_month"],
        "on_time_rate_by_impact": on_time_by_impact.round(4).to_dict(),
    }
    (BASE / "reports" / "monitoring_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
