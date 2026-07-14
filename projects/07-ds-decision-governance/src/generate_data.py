"""
Synthetic data generator for a data science team's decision-governance
log: one row per recorded decision (a model launch, an experiment
rollout, a dashboard change, a pipeline change, a metric-definition
change, or a deprecation), with an impact level, an approval lag, and
two follow-up monitoring commitments (a ship check and a longer-horizon
metric check) that are either kept on time, kept late, or missed.

All data is synthetic. Relationships between impact level and approval
speed, monitoring follow-through, and rollback rate are hand-specified
below (with noise) to resemble a plausible governance process, not fit
from any real team's records. A short capacity-crunch window is also
injected (fewer checks land on time for a few months), the same role
project 04's synthetic quality regression plays for its control chart.
Fields are generated first, independent of the eventual outcome, and
outcome is computed from them afterward, the same generation order used
in every other project in this repo.

Run:
    python src/generate_data.py
Writes:
    data/decision_log.csv (900 decision records over a 2-year window)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_RECORDS = 900
N_DAYS = 730  # ~2-year window
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
EPOCH = pd.Timestamp("2024-01-01")

rng = np.random.default_rng(SEED)

ARTIFACT_TYPES = [
    "dashboard_change", "pipeline_change", "experiment_rollout",
    "model_launch", "metric_definition_change", "deprecation",
]
ARTIFACT_TYPE_WEIGHTS = [0.30, 0.22, 0.20, 0.14, 0.09, 0.05]

# P(low), P(medium), P(high) impact, by artifact type: a dashboard tweak
# is usually low-stakes; a model launch or a deprecation is more often
# medium or high, since it can change what a downstream team depends on.
IMPACT_WEIGHTS_BY_ARTIFACT = {
    "dashboard_change": [0.75, 0.20, 0.05],
    "pipeline_change": [0.45, 0.40, 0.15],
    "experiment_rollout": [0.55, 0.35, 0.10],
    "model_launch": [0.15, 0.45, 0.40],
    "metric_definition_change": [0.30, 0.45, 0.25],
    "deprecation": [0.20, 0.40, 0.40],
}
IMPACT_LEVELS = ["low", "medium", "high"]

DOMAIN_TAGS = ["product_analytics", "search_ranking", "marketing", "customer_support", "operations", "infrastructure"]
DOMAIN_WEIGHTS = [0.22, 0.18, 0.15, 0.15, 0.15, 0.15]

# Approval takes longer, and is more likely to stall before ever getting
# approved, the higher the impact level: more reviewers, more scrutiny.
APPROVAL_LAG_MEAN_DAYS = {"low": 1.5, "medium": 5.0, "high": 13.0}
ABANDON_PROB = {"low": 0.03, "medium": 0.06, "high": 0.12}

SHIP_LAG_MEAN_DAYS = 4.0  # approval -> shipped, roughly independent of impact level

SHIP_CHECK_WINDOW_DAYS = 7     # short follow-up: did it ship as intended?
METRIC_CHECK_WINDOW_DAYS = 30  # longer follow-up: did it actually work?

# Baseline probability a monitoring check gets closed by its due date.
# Lower-impact items get less attention day to day, so they're the ones
# most likely to slip.
ON_TIME_BASE_PROB = {"low": 0.55, "medium": 0.75, "high": 0.90}

# A capacity-crunch window: for about three months, check follow-through
# drops across the board, regardless of impact level. This is the signal
# the control chart in open_loops.py is built to catch.
CRUNCH_MONTH_START = 13
CRUNCH_MONTH_END = 15
CRUNCH_ON_TIME_MULTIPLIER = 0.55

# Once a metric check happens, the decision is kept, iterated on, or
# rolled back. Higher-impact decisions get more scrutiny before they
# ship, so they're less likely to end up rolled back after the fact.
OUTCOME_PROBS = {
    "low": {"keep": 0.62, "iterate": 0.20, "rollback": 0.18},
    "medium": {"keep": 0.70, "iterate": 0.20, "rollback": 0.10},
    "high": {"keep": 0.75, "iterate": 0.20, "rollback": 0.05},
}


def make_decisions(n=N_RECORDS):
    artifact_type = rng.choice(ARTIFACT_TYPES, size=n, p=ARTIFACT_TYPE_WEIGHTS)
    domain_tag = rng.choice(DOMAIN_TAGS, size=n, p=DOMAIN_WEIGHTS)
    impact_level = np.array([
        rng.choice(IMPACT_LEVELS, p=IMPACT_WEIGHTS_BY_ARTIFACT[a]) for a in artifact_type
    ])
    proposed_day_offset = rng.integers(0, N_DAYS, size=n)

    return pd.DataFrame({
        "decision_id": np.arange(1, n + 1),
        "artifact_type": artifact_type,
        "domain_tag": domain_tag,
        "impact_level": pd.Categorical(impact_level, categories=IMPACT_LEVELS, ordered=True),
        "proposed_date": EPOCH + pd.to_timedelta(proposed_day_offset, unit="D"),
    })


def add_approval_outcome(df):
    """Whether, and how fast, each decision gets approved."""
    df = df.copy()
    abandon_prob = df["impact_level"].map(ABANDON_PROB).astype(float)
    df["abandoned"] = rng.uniform(size=len(df)) < abandon_prob

    lag_mean = df["impact_level"].map(APPROVAL_LAG_MEAN_DAYS).astype(float)
    approval_lag_days = rng.lognormal(mean=np.log(lag_mean), sigma=0.55)
    df["approval_lag_days"] = np.where(df["abandoned"], np.nan, approval_lag_days.round(1))
    df["approved_date"] = np.where(
        df["abandoned"], pd.NaT,
        df["proposed_date"] + pd.to_timedelta(df["approval_lag_days"].round(0), unit="D"),
    )
    df["approved_date"] = pd.to_datetime(df["approved_date"])
    return df


def add_shipping(df):
    """Approved decisions ship a few days after approval."""
    df = df.copy()
    ship_lag_days = rng.lognormal(mean=np.log(SHIP_LAG_MEAN_DAYS), sigma=0.4, size=len(df))
    df["shipped_date"] = np.where(
        df["abandoned"], pd.NaT,
        df["approved_date"] + pd.to_timedelta(ship_lag_days.round(0), unit="D"),
    )
    df["shipped_date"] = pd.to_datetime(df["shipped_date"])
    return df


def _month_index(dates):
    """Whole months since EPOCH, for bucketing due dates into the
    capacity-crunch window."""
    return (dates.dt.year - EPOCH.year) * 12 + (dates.dt.month - EPOCH.month)


def _on_time_draw(due_dates, impact_level):
    base_prob = impact_level.map(ON_TIME_BASE_PROB).astype(float).to_numpy()
    in_crunch = _month_index(due_dates).between(CRUNCH_MONTH_START, CRUNCH_MONTH_END).to_numpy()
    prob = np.where(in_crunch, base_prob * CRUNCH_ON_TIME_MULTIPLIER, base_prob)
    return rng.uniform(size=len(due_dates)) < prob


def add_monitoring_checks(df):
    """Two follow-up commitments made at ship time: a short ship check
    (did it ship as intended) required only for medium/high impact
    decisions, and a longer metric check (did it actually work) required
    for every shipped decision. Each either closes on time or slips."""
    df = df.copy()
    shipped = ~df["abandoned"]

    df["ship_check_required"] = shipped & df["impact_level"].isin(["medium", "high"])
    df["ship_check_due"] = pd.NaT
    df.loc[df["ship_check_required"], "ship_check_due"] = (
        df.loc[df["ship_check_required"], "shipped_date"]
        + pd.to_timedelta(SHIP_CHECK_WINDOW_DAYS, unit="D")
    )
    df["ship_check_due"] = pd.to_datetime(df["ship_check_due"])
    df["ship_check_on_time"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    req = df["ship_check_required"]
    df.loc[req, "ship_check_on_time"] = _on_time_draw(df.loc[req, "ship_check_due"], df.loc[req, "impact_level"])

    df["metric_check_due"] = pd.NaT
    df.loc[shipped, "metric_check_due"] = (
        df.loc[shipped, "shipped_date"] + pd.to_timedelta(METRIC_CHECK_WINDOW_DAYS, unit="D")
    )
    df["metric_check_due"] = pd.to_datetime(df["metric_check_due"])
    df["metric_check_on_time"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[shipped, "metric_check_on_time"] = _on_time_draw(
        df.loc[shipped, "metric_check_due"], df.loc[shipped, "impact_level"]
    )
    return df


def assign_outcome(df):
    """Keep / iterate / rollback, drawn once the metric check happens,
    plus the final status field derived from abandonment and outcome."""
    df = df.copy()
    shipped = ~df["abandoned"]
    outcome = np.full(len(df), None, dtype=object)
    for level in IMPACT_LEVELS:
        mask = shipped & (df["impact_level"] == level)
        probs = OUTCOME_PROBS[level]
        outcome[mask.to_numpy()] = rng.choice(
            list(probs.keys()), size=int(mask.sum()), p=list(probs.values())
        )
    df["outcome"] = outcome

    status = np.select(
        [df["abandoned"], df["outcome"] == "rollback", shipped],
        ["abandoned", "reverted", "closed"],
        default="abandoned",
    )
    df["status"] = status
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_decisions(N_RECORDS)
    df = add_approval_outcome(df)
    df = add_shipping(df)
    df = add_monitoring_checks(df)
    df = assign_outcome(df)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_path = OUT_DIR / "decision_log.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} decision records -> {out_path}")
    print(f"Status mix:\n{df['status'].value_counts(normalize=True).round(3)}")
    print(f"Approval lag (days), mean by impact level:\n{df.groupby('impact_level', observed=True)['approval_lag_days'].mean().round(2)}")
    print(f"Rollback rate among closed/reverted, by impact level:\n"
          f"{df[df['status'].isin(['closed', 'reverted'])].groupby('impact_level', observed=True)['outcome'].apply(lambda s: (s == 'rollback').mean()).round(3)}")
    print(f"Metric-check on-time rate, by impact level:\n"
          f"{df[df['metric_check_on_time'].notna()].groupby('impact_level', observed=True)['metric_check_on_time'].mean().round(3)}")


if __name__ == "__main__":
    main()
