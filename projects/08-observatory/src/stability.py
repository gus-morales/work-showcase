"""
Wraps popmon to answer, for a batch of model features observed over
time, whether any of them drifted. Where the ops-metric detectors
(src/detectors.py) each check one scalar series against its own
history, popmon compares the full distribution of each feature, day
over day, against a reference period, catching a shifted mix or a
level change that a scalar check never sees, since there's no single
number to check in the first place.

run_stability_report() is the only thing that runs popmon itself;
extract_alerts() turns its traffic-light output into the exact same
long-form shape detectors.run_detectors() produces (day, metric,
detector, flagged), so both engines feed the same
alerts.find_alert_events() unchanged.
"""
import pandas as pd

import popmon  # noqa: F401  (registers df.pm_stability_report)

DETECTOR_NAME = "popmon"

# popmon runs somewhere between 15 and 30 individual statistical checks
# per feature per day (profile stats, histogram comparisons, trend
# significance, ...); a handful landing on red on any given day is
# normal background noise, not every check agrees with every other one
# even when nothing is wrong. REFERENCE_DAYS establishes what "normal"
# actually looks like for each feature before anything is flagged, the
# same idea as a p-chart's reference period (see project 04). Passing
# it to popmon explicitly matters more here than it would for a sharp
# level shift: popmon's default bounds are computed from the whole
# series' own variability, so a slow, continuous drift widens its own
# comparison bounds as it happens and can end up looking like normal
# variation unless it's compared against a reference period the drift
# never touches.
REFERENCE_DAYS = range(0, 15)
RED_MARGIN = 2


def run_stability_report(df: pd.DataFrame, time_axis: str, features: list[str], time_width: str = "1d",
                          reference_days: range = REFERENCE_DAYS):
    """features: catalog feature names, without the time-axis prefix;
    popmon wants each one as f'{time_axis}:{feature}'. Compares every
    period against reference_days explicitly, not against the whole
    series (see REFERENCE_DAYS above)."""
    popmon_features = [f"{time_axis}:{feature}" for feature in features]
    reference = df[df["day"] < reference_days.stop]
    return df.pm_stability_report(time_axis=time_axis, features=popmon_features, time_width=time_width,
                                   reference=reference)


def extract_alerts(
    report,
    start_date: pd.Timestamp,
    reference_days: range = REFERENCE_DAYS,
    red_margin: int = RED_MARGIN,
) -> pd.DataFrame:
    """Turn popmon's per-feature red/yellow/green counts per time bin
    into long-form (day, metric, detector, flagged). A day is flagged
    only if its red-check count is meaningfully above that feature's
    own reference-period baseline (red_margin higher than the median
    red count over reference_days), not merely nonzero. Days inside
    the reference window itself are never flagged, there isn't enough
    history yet to say whether they're normal."""
    alerts = report.datastore["alerts"]
    rows = []
    for feature, counts in alerts.items():
        if feature == "_AGGREGATE_":
            continue
        counts = counts.copy()
        counts["day"] = [
            int((pd.Timestamp(ts).normalize() - start_date) / pd.Timedelta(days=1))
            for ts in counts.index
        ]
        reference = counts[counts["day"].isin(reference_days)]
        baseline = reference["n_red"].median() if len(reference) else 0
        cutoff = baseline + red_margin

        for _, row in counts.iterrows():
            flagged = row["day"] >= reference_days.stop and row["n_red"] >= cutoff
            rows.append({
                "day": row["day"],
                "metric": feature,
                "detector": DETECTOR_NAME,
                "flagged": bool(flagged),
            })
    return pd.DataFrame(rows, columns=["day", "metric", "detector", "flagged"])
