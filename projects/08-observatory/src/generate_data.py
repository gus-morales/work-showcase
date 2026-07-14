"""
Synthetic data for project 08: two datasets behind one DuckDB database,
one for each detection engine. Domain-agnostic DS-team operations, not
tied to any specific employer: a nightly model-training pipeline, and
the churn model it trains, scoring a batch of customers every day.

pipeline_runs: one row per day, four operational metrics, each with a
deliberately different kind of injected anomaly, so each needs a
different detector (src/detectors.py) to catch cleanly:
- pipeline_duration_minutes: a short spike (threshold breach)
- pipeline_success_rate: a sustained step change (statistical outlier)
- data_freshness_hours: a gradual increase (trend break)
- row_count: a reporting gap (missing data)

scoring_log: one row per customer scored per day, five churn-model
features, four monitored for drift and one left stable as a control,
run through popmon instead (src/stability.py), since population and
feature-distribution shift isn't the kind of anomaly a scalar detector
is built to catch:
- monthly_usage_score: a level shift
- plan_tier: a categorical mix shift
- support_tickets_30d: a missing-data spike
- predicted_churn_prob: a gradual drift (the model's own output)
- tenure_months: left alone, a stable reference feature

Run:
    python src/generate_data.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

SEED = 13
rng = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = OUT_DIR / "observatory.duckdb"

N_DAYS = 90
START_DATE = pd.Timestamp("2026-01-01")
ROWS_PER_DAY = 180
PLAN_TIERS = np.array(["basic", "pro", "enterprise"])


def make_pipeline_runs(n_days: int = N_DAYS) -> pd.DataFrame:
    days = np.arange(n_days)
    dates = START_DATE + pd.to_timedelta(days, unit="D")

    duration = rng.normal(45, 4, size=n_days)
    duration[30:33] = rng.normal(180, 15, size=3)

    success_rate = rng.normal(0.98, 0.006, size=n_days)
    success_rate[55:] = rng.normal(0.85, 0.008, size=n_days - 55)

    freshness = rng.normal(2.0, 0.3, size=n_days)
    freshness[65:80] = 2.0 + np.linspace(0, 7.0, 15)
    freshness[80:] = 9.0 + rng.normal(0, 0.3, size=n_days - 80)

    row_count = rng.normal(500, 40, size=n_days)

    df = pd.DataFrame({
        "day": days,
        "date": dates,
        "pipeline_duration_minutes": duration.round(1),
        "pipeline_success_rate": success_rate.clip(0, 1).round(4),
        "data_freshness_hours": freshness.clip(0).round(2),
        "row_count": row_count.round(0).astype("float64"),
    })
    df.loc[40:42, "row_count"] = np.nan
    return df


def make_scoring_log(n_days: int = N_DAYS, rows_per_day: int = ROWS_PER_DAY) -> pd.DataFrame:
    rows = []
    for day in range(n_days):
        date = START_DATE + pd.Timedelta(days=day)
        n = rows_per_day

        tenure = rng.gamma(shape=2.0, scale=10.0, size=n).round(1)

        usage_mean = 55.0 if day < 50 else 30.0
        usage = rng.normal(usage_mean, 8.0, size=n).clip(0, 100).round(1)

        plan_probs = [0.50, 0.35, 0.15] if day < 60 else [0.15, 0.30, 0.55]
        plan_tier = rng.choice(PLAN_TIERS, size=n, p=plan_probs)

        tickets = rng.poisson(1.2, size=n).astype(float)
        missing_rate = 0.70 if 70 <= day <= 73 else 0.15
        tickets[rng.random(n) < missing_rate] = np.nan

        drift_progress = np.clip((day - 20) / 60, 0, 1)
        churn_mean = 0.15 + drift_progress * 0.20
        churn_prob = rng.normal(churn_mean, 0.04, size=n).clip(0.01, 0.99).round(4)

        for i in range(n):
            rows.append({
                "day": day, "date": date, "customer_id": f"C{day:03d}{i:04d}",
                "tenure_months": tenure[i], "monthly_usage_score": usage[i],
                "support_tickets_30d": tickets[i], "plan_tier": plan_tier[i],
                "predicted_churn_prob": churn_prob[i],
            })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_runs = make_pipeline_runs()
    scoring_log = make_scoring_log()

    con = duckdb.connect(str(DB_PATH))
    con.register("pipeline_runs_view", pipeline_runs)
    con.execute("CREATE OR REPLACE TABLE pipeline_runs AS SELECT * FROM pipeline_runs_view")
    con.unregister("pipeline_runs_view")

    con.register("scoring_log_view", scoring_log)
    con.execute("CREATE OR REPLACE TABLE scoring_log AS SELECT * FROM scoring_log_view")
    con.unregister("scoring_log_view")
    con.close()

    print(f"Wrote {len(pipeline_runs)} pipeline_runs rows and {len(scoring_log):,} scoring_log rows "
          f"-> data/observatory.duckdb")
    print("Ops injections: duration spike (days 30-32), success-rate step change (day 55+), "
          "freshness ramp (days 65-79), row-count gap (days 40-42)")
    print("Model injections: usage-score shift (day 50+), plan-tier mix shift (day 60+), "
          "ticket-count reporting gap (days 70-73), churn-prob drift (days 20-80)")


if __name__ == "__main__":
    main()
