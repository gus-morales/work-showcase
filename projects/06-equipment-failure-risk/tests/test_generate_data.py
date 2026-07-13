"""Data-generation invariants: schema, ranges, and no-null checks on a
small synthetic sample. Not testing exact values (the generator is
stochastic by design), just the contract the rest of the pipeline
relies on, plus the directional relationships the failure label is
supposed to encode (overdue maintenance, fault codes, vibration, low
oil pressure, and equipment age should all raise the failure rate,
since those are exactly the signals the model in train.py is supposed
to recover in interpret.py)."""
import pytest

import generate_data as gd


@pytest.fixture(scope="module")
def trucks():
    return gd.make_trucks(40)


@pytest.fixture(scope="module")
def panel(trucks):
    df = gd.make_daily_panel(trucks)
    df = gd.add_maintenance_schedule(df)
    df = gd.add_condition_signals(df)
    return df


@pytest.fixture(scope="module")
def full(panel):
    return gd.assign_failure_label(panel)


def test_trucks_schema_and_ranges(trucks):
    expected_cols = {
        "truck_id", "truck_class", "site", "equipment_age_years_at_start",
        "wear_rate_factor", "daily_usage_hours_mean",
    }
    assert expected_cols.issubset(trucks.columns)
    assert trucks["truck_id"].is_unique
    assert trucks["truck_class"].isin(gd.TRUCK_CLASSES).all()
    assert trucks["site"].isin(gd.SITES).all()
    assert trucks["equipment_age_years_at_start"].between(0, 18).all()
    assert trucks["daily_usage_hours_mean"].between(8, 20).all()
    assert not trucks.isnull().any().any()


def test_panel_has_one_row_per_truck_per_day(trucks):
    df = gd.make_daily_panel(trucks)
    assert len(df) == len(trucks) * gd.N_DAYS
    assert set(df["truck_id"]) == set(trucks["truck_id"])


def test_panel_schema_and_ranges(panel):
    assert panel["engine_hours_at_obs"].ge(0).all()
    assert panel["payload_avg_tons"].between(50, 450).all()
    assert panel["downtime_cost_usd"].between(5_000, 180_000).all()
    assert panel["days_since_last_pm"].ge(0).all()
    assert panel["vibration_rms_mm_s"].between(0.5, 20).all()
    assert panel["oil_pressure_psi"].between(15, 75).all()
    assert panel["coolant_temp_c"].between(65, 118).all()
    assert panel["brake_wear_pct"].between(0, 100).all()
    assert (panel["fault_codes_7d"] >= 0).all()
    assert not panel.isnull().any().any()


def test_failure_flag_is_binary(full):
    assert full["failure_within_7d"].isin([0, 1]).all()


def test_failure_rate_is_plausible(full):
    rate = full["failure_within_7d"].mean()
    # Loose sanity band: catches a broken generator (inverted sign,
    # runaway intercept) without being flaky about the exact rate.
    assert 0.003 < rate < 0.06


def test_overdue_maintenance_raises_failure_rate(full):
    overdue_rate = full.loc[full.days_since_last_pm >= 45, "failure_within_7d"].mean()
    current_rate = full.loc[full.days_since_last_pm < 15, "failure_within_7d"].mean()
    assert overdue_rate > current_rate


def test_recent_fault_codes_raise_failure_rate(full):
    busy_rate = full.loc[full.fault_codes_7d >= 1, "failure_within_7d"].mean()
    quiet_rate = full.loc[full.fault_codes_7d == 0, "failure_within_7d"].mean()
    assert busy_rate > quiet_rate


def test_high_vibration_raises_failure_rate(full):
    high = full.loc[full.vibration_rms_mm_s >= full.vibration_rms_mm_s.median(), "failure_within_7d"].mean()
    low = full.loc[full.vibration_rms_mm_s < full.vibration_rms_mm_s.median(), "failure_within_7d"].mean()
    assert high > low


def test_low_oil_pressure_raises_failure_rate(full):
    low_pressure = full.loc[full.oil_pressure_psi < full.oil_pressure_psi.median(), "failure_within_7d"].mean()
    high_pressure = full.loc[full.oil_pressure_psi >= full.oil_pressure_psi.median(), "failure_within_7d"].mean()
    assert low_pressure > high_pressure


def test_older_equipment_has_higher_failure_rate(full):
    old_rate = full.loc[full.equipment_age_years_at_obs > 10, "failure_within_7d"].mean()
    new_rate = full.loc[full.equipment_age_years_at_obs <= 2, "failure_within_7d"].mean()
    assert old_rate > new_rate
