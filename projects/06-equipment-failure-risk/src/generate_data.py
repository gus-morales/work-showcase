"""
Synthetic data generator for a mining haul-truck predictive-maintenance
dataset: a fictional open-pit mining operation, viewed from the
fleet-maintenance side. Truck-day telemetry records (engine hours,
vibration, oil pressure, coolant temperature, brake wear, fault codes,
overdue-maintenance signal) and an unplanned-failure outcome
(`failure_within_7d`) at a realistic, heavily imbalanced rate.

All data is synthetic. Relationships between features and the failure
outcome are hand-specified below (with noise) to resemble plausible
predictive-maintenance drivers (rising vibration, falling oil pressure,
rising coolant temperature, brake wear, recent fault codes, days
overdue for preventive maintenance, equipment age), not fit from any
real fleet. Features are generated first, independent of the failure
outcome, and the label is computed from them afterward, the same
generation order used in every other project in this repo, so that
"the model recovers known relationships" in the interpretability
section is a genuine check rather than a foregone conclusion.

Run:
    python src/generate_data.py
Writes:
    data/truck_telemetry.csv (150 trucks x 600 days = 90k truck-day records)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_TRUCKS = 150
N_DAYS = 600
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

rng = np.random.default_rng(SEED)

TRUCK_CLASSES = ["ultra_class_400t", "large_class_300t", "standard_class_200t"]
TRUCK_CLASS_WEIGHTS = [0.25, 0.40, 0.35]
# Baseline payload (tons) and estimated production-loss cost per unplanned
# failure event, by truck class: a bigger truck moves more ore per cycle,
# so an unplanned failure on one is worth more lost production.
TRUCK_CLASS_PAYLOAD_TONS = {"ultra_class_400t": 380, "large_class_300t": 280, "standard_class_200t": 190}
TRUCK_CLASS_DOWNTIME_BASE_USD = {"ultra_class_400t": 60_000, "large_class_300t": 42_000, "standard_class_200t": 26_000}

SITES = ["north_pit", "south_pit", "east_pit", "ridge_extension"]
SITE_WEIGHTS = [0.32, 0.28, 0.22, 0.18]
SITE_BASE_AMBIENT_C = {"north_pit": 24, "south_pit": 31, "east_pit": 27, "ridge_extension": 19}

PM_INTERVAL_MEAN_DAYS = 45  # typical scheduled preventive-maintenance interval
INTERCEPT = -7.65  # calibrated to a ~1.8% base failure rate


def make_trucks(n=N_TRUCKS):
    truck_id = np.arange(1, n + 1)
    truck_class = rng.choice(TRUCK_CLASSES, size=n, p=TRUCK_CLASS_WEIGHTS)
    site = rng.choice(SITES, size=n, p=SITE_WEIGHTS)
    equipment_age_years_at_start = np.clip(rng.exponential(scale=4.5, size=n), 0, 18)
    # Per-truck wear-rate multiplier (a persistent random effect, the same
    # role project 01/05's "spend_scale" plays): trucks with a harder duty
    # cycle or a weaker maintenance history degrade faster than average
    # between services, independent of how many days it's been.
    wear_rate_factor = rng.gamma(shape=4.0, scale=0.25, size=n)
    daily_usage_hours_mean = np.clip(rng.normal(15, 2, size=n), 8, 20)

    return pd.DataFrame({
        "truck_id": truck_id,
        "truck_class": truck_class,
        "site": site,
        "equipment_age_years_at_start": equipment_age_years_at_start,
        "wear_rate_factor": wear_rate_factor,
        "daily_usage_hours_mean": daily_usage_hours_mean,
    })


def make_daily_panel(trucks):
    """Full truck x day grid: every truck observed once per day over the
    whole window, unlike project 05's variable per-customer transaction
    count, since a fleet-telemetry feed is a regular daily read rather
    than an event stream."""
    n = len(trucks)
    day_offset = np.tile(np.arange(N_DAYS), n)
    rows = trucks.loc[trucks.index.repeat(N_DAYS)].reset_index(drop=True)
    rows["day_offset"] = day_offset

    epoch = pd.Timestamp("2024-01-01")
    rows["date"] = epoch + pd.to_timedelta(rows["day_offset"], unit="D")
    rows["equipment_age_years_at_obs"] = rows["equipment_age_years_at_start"] + rows["day_offset"] / 365.0
    rows["engine_hours_at_obs"] = (
        rows["equipment_age_years_at_start"] * 4000
        + rows["day_offset"] * rows["daily_usage_hours_mean"]
    ).round(0).astype(int)

    rows["ambient_temp_c"] = rows["site"].map(SITE_BASE_AMBIENT_C) + rng.normal(0, 5, size=len(rows))

    payload_base = rows["truck_class"].map(TRUCK_CLASS_PAYLOAD_TONS)
    rows["payload_avg_tons"] = np.clip(payload_base * rng.lognormal(0, 0.08, size=len(rows)), 50, 450).round(1)

    downtime_base = rows["truck_class"].map(TRUCK_CLASS_DOWNTIME_BASE_USD)
    rows["downtime_cost_usd"] = np.clip(
        downtime_base * rng.lognormal(mean=0.0, sigma=0.45, size=len(rows)), 5_000, 180_000
    ).round(2)

    return rows


def add_maintenance_schedule(df):
    """Days since the last scheduled preventive-maintenance (PM) event, per
    truck. PM events follow a renewal process (interval ~45 days, jittered),
    independent of the failure outcome, since a maintenance calendar is
    set on its own schedule, not in response to failures that haven't
    happened yet. Computed with a per-truck asof join against each truck's
    PM-event days, the same style of vectorized per-group lookup project
    05 uses for its rolling velocity windows."""
    pm_rows = []
    for truck_id in df["truck_id"].unique():
        day = rng.uniform(0, PM_INTERVAL_MEAN_DAYS)  # first PM isn't synchronized across trucks
        while day < N_DAYS:
            pm_rows.append((truck_id, round(day)))
            day += np.clip(rng.exponential(PM_INTERVAL_MEAN_DAYS), 20, 90)
    pm_df = pd.DataFrame(pm_rows, columns=["truck_id", "pm_day"]).sort_values(["pm_day", "truck_id"])
    pm_df["pm_day"] = pm_df["pm_day"].astype(float)

    left = df.sort_values(["day_offset", "truck_id"]).reset_index(drop=True)
    left["day_offset"] = left["day_offset"].astype(float)
    merged = pd.merge_asof(
        left, pm_df, left_on="day_offset", right_on="pm_day", by="truck_id", direction="backward",
    )
    merged["days_since_last_pm"] = (merged["day_offset"] - merged["pm_day"].fillna(0)).clip(lower=0).astype(int)
    return merged.drop(columns=["pm_day"])


def add_condition_signals(df):
    """Sensor/inspection signals that degrade with days overdue for
    maintenance, scaled by each truck's own wear-rate factor, plus
    independent noise. Values are plausible ranges for heavy mining
    haul trucks, not manufacturer specifications."""
    overdue = df["days_since_last_pm"] * df["wear_rate_factor"]

    df["vibration_rms_mm_s"] = np.clip(
        rng.lognormal(mean=np.log(2.5), sigma=0.30, size=len(df)) * (1 + 0.018 * overdue), 0.5, 20
    ).round(3)

    df["oil_pressure_psi"] = np.clip(
        rng.normal(58, 4, size=len(df)) - 0.045 * overdue, 15, 75
    ).round(1)

    df["coolant_temp_c"] = np.clip(
        rng.normal(87, 3, size=len(df)) + 0.028 * overdue + 0.15 * (df["ambient_temp_c"] - 25), 65, 118
    ).round(1)

    df["brake_wear_pct"] = np.clip(
        5 + 0.075 * overdue + rng.normal(0, 5, size=len(df)), 0, 100
    ).round(1)

    fault_code_rate = np.clip(0.08 + 0.0055 * overdue, 0.03, 4.0)
    df["fault_codes_7d"] = rng.poisson(fault_code_rate)

    return df


def assign_failure_label(df):
    z = (
        0.55 * (df["vibration_rms_mm_s"] - 2.5)
        - 0.085 * (df["oil_pressure_psi"] - 55)
        + 0.10 * (df["coolant_temp_c"] - 88)
        + 0.026 * df["brake_wear_pct"]
        + 0.50 * df["fault_codes_7d"]
        + 0.017 * df["days_since_last_pm"]
        + 0.085 * df["equipment_age_years_at_obs"]
        + INTERCEPT
        + rng.normal(0, 1.0, size=len(df))
    )
    prob = 1 / (1 + np.exp(-z))
    df = df.copy()
    df["failure_within_7d"] = rng.binomial(1, prob)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trucks = make_trucks(N_TRUCKS)
    panel = make_daily_panel(trucks)
    panel = add_maintenance_schedule(panel)
    panel = add_condition_signals(panel)
    full = assign_failure_label(panel)
    full = full.drop(columns=["equipment_age_years_at_start", "wear_rate_factor", "daily_usage_hours_mean", "day_offset"])
    full = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_path = OUT_DIR / "truck_telemetry.csv"
    full.to_csv(out_path, index=False)
    print(f"Wrote {len(full):,} truck-day records across {full['truck_id'].nunique():,} trucks -> {out_path}")
    print(f"Overall unplanned-failure rate: {full['failure_within_7d'].mean():.3%}")
    print(f"Failure rate, days_since_last_pm >= 60 vs. < 15: "
          f"{full.loc[full.days_since_last_pm >= 60, 'failure_within_7d'].mean():.3%} vs. "
          f"{full.loc[full.days_since_last_pm < 15, 'failure_within_7d'].mean():.3%}")
    print(f"Failure rate, fault_codes_7d >= 2 vs. 0: "
          f"{full.loc[full.fault_codes_7d >= 2, 'failure_within_7d'].mean():.3%} vs. "
          f"{full.loc[full.fault_codes_7d == 0, 'failure_within_7d'].mean():.3%}")


if __name__ == "__main__":
    main()
