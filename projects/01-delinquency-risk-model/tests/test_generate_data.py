"""Data-generation invariants: schema, ranges, and no-null checks on a
small synthetic sample. Not testing exact values (the generator is
stochastic by design), just the contract the rest of the pipeline
relies on."""
import pytest

import generate_data as gd


@pytest.fixture(scope="module")
def customers():
    # Large enough that the shock-window comparison below isn't noise-dominated
    # (the shock window is only the last 2 of 24 months of originations).
    return gd.make_customers(3000)


@pytest.fixture(scope="module")
def loans(customers):
    return gd.make_loans(customers)


@pytest.fixture(scope="module")
def full(loans, customers):
    return gd.assign_delinquency(loans, customers)


@pytest.fixture(scope="module")
def with_survival(full):
    return gd.add_survival_columns(full)


def test_customers_schema_and_ranges(customers):
    expected_cols = {
        "customer_id", "age", "city", "city_tier", "employment_type",
        "monthly_income_usd", "tenure_months_platform", "num_previous_loans",
        "credit_bureau_score", "avg_prior_repayment_delay_days",
        "num_active_loans_elsewhere", "device_type", "acquisition_channel",
    }
    assert expected_cols.issubset(customers.columns)
    assert len(customers) == 3000
    assert customers["customer_id"].is_unique
    assert customers["age"].between(18, 70).all()
    assert customers["credit_bureau_score"].between(300, 850).all()
    assert customers["monthly_income_usd"].gt(0).all()
    assert customers["city_tier"].isin(["tier1", "tier2", "tier3"]).all()
    assert not customers.isnull().any().any()


def test_income_increases_with_city_tier(customers):
    # Tier 1 metros should carry a higher mean income than tier 3, since
    # the multiplier is hard-coded that way; this is the one thing worth
    # locking in given it drives loan_to_income_ratio downstream.
    means = customers.groupby("city_tier")["monthly_income_usd"].mean()
    assert means["tier1"] > means["tier2"] > means["tier3"]


def test_loans_schema_and_ranges(loans):
    assert loans["loan_id"].is_unique
    assert loans["loan_amount_usd"].between(300, 25_000).all()
    assert loans["num_installments"].isin(gd.INSTALLMENT_OPTIONS).all()
    assert loans["down_payment_ratio"].between(0, 0.5).all()
    assert loans["origination_month"].between(1, gd.N_MONTHS).all()
    assert not loans.isnull().any().any()


def test_loans_reference_valid_customers(loans, customers):
    assert set(loans["customer_id"]).issubset(set(customers["customer_id"]))


def test_delinquency_flag_is_binary(full):
    assert full["delinquent_30dpd"].isin([0, 1]).all()


def test_delinquency_rate_is_plausible(full):
    rate = full["delinquent_30dpd"].mean()
    # Loose sanity band, not a tight statistical assertion: catches a
    # broken generator (e.g. an inverted sign) without being flaky.
    assert 0.02 < rate < 0.40


def test_shock_window_raises_delinquency(full):
    shocked = full[full.origination_month >= gd.N_MONTHS - 2]
    normal = full[full.origination_month < gd.N_MONTHS - 2]
    assert shocked["delinquent_30dpd"].mean() > normal["delinquent_30dpd"].mean()


def test_bureau_missingness_does_not_change_delinquency_labels(full):
    # Masking credit_bureau_score to NaN happens strictly after labels are
    # assigned, using the same rows/order, so the labels themselves must
    # be untouched by it.
    masked = gd.apply_bureau_missingness(full)
    assert (masked["delinquent_30dpd"] == full["delinquent_30dpd"]).all()


def test_bureau_missingness_rate_is_plausible(full):
    masked = gd.apply_bureau_missingness(full)
    rate = masked["credit_bureau_score"].isna().mean()
    assert 0.03 < rate < 0.25


def test_bureau_missingness_concentrates_in_thin_file_segments(full):
    masked = gd.apply_bureau_missingness(full)
    informal_rate = masked.loc[masked.employment_type == "informal", "credit_bureau_score"].isna().mean()
    salaried_rate = masked.loc[masked.employment_type == "salaried", "credit_bureau_score"].isna().mean()
    assert informal_rate > salaried_rate


def test_survival_columns_do_not_change_delinquency_labels(full, with_survival):
    # add_survival_columns draws from its own isolated RNG stream and must
    # not touch the already-assigned binary label.
    assert (with_survival["delinquent_30dpd"] == full["delinquent_30dpd"]).all()


def test_event_observed_matches_delinquent_flag(with_survival):
    assert (with_survival["event_observed"] == with_survival["delinquent_30dpd"]).all()


def test_time_to_30dpd_is_positive_and_bounded(with_survival):
    assert (with_survival["time_to_30dpd_days"] > 0).all()
    loan_term_days = with_survival["num_installments"] * 30
    assert (with_survival["time_to_30dpd_days"] <= loan_term_days).all()


def test_events_fail_faster_than_censored_loans(with_survival):
    # Loans that actually went delinquent should, on average, have a much
    # shorter recorded time-to-event than loans that were censored, since
    # the hazard is parameterized by the same risk score that drives
    # whether the event happens at all.
    event_mean = with_survival.loc[with_survival.event_observed == 1, "time_to_30dpd_days"].mean()
    censored_mean = with_survival.loc[with_survival.event_observed == 0, "time_to_30dpd_days"].mean()
    assert event_mean < censored_mean
