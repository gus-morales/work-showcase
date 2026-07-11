"""
Data contracts: schema and range checks the generated customers.csv
and orders.csv must satisfy before anything downstream trusts them.
Runs at the end of generate_data.py (so a broken generator fails
loudly instead of quietly writing bad data) and again in db.py's
get_connection() (so a stale or hand-edited CSV can't silently break
every script that reads it instead).
"""
import pandas as pd

N_MONTHS = 24  # kept in sync with generate_data.py's N_MONTHS

VALID_CITY_TIERS = {"tier1", "tier2", "tier3"}
VALID_EMPLOYMENT_TYPES = {"salaried", "self_employed", "gig_economy", "informal"}
VALID_CHANNELS = {"organic", "referral", "partner_store", "paid_social"}


class DataContractError(ValueError):
    """Raised when a dataset fails its contract."""


def validate_customers(df: pd.DataFrame) -> list[str]:
    violations = []
    required = ["customer_id", "cohort_month", "acquisition_channel", "city_tier",
                "employment_type", "mean_order_value_usd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"customers: missing required column '{c}'" for c in missing]

    if df["customer_id"].isna().any():
        violations.append("customers: customer_id has nulls")
    if df["customer_id"].duplicated().any():
        violations.append("customers: customer_id has duplicates")
    if not df["cohort_month"].between(1, N_MONTHS).all():
        violations.append(f"customers: cohort_month outside [1, {N_MONTHS}]")

    bad_channels = set(df["acquisition_channel"].unique()) - VALID_CHANNELS
    if bad_channels:
        violations.append(f"customers: unexpected acquisition_channel values {sorted(bad_channels)}")
    bad_tiers = set(df["city_tier"].unique()) - VALID_CITY_TIERS
    if bad_tiers:
        violations.append(f"customers: unexpected city_tier values {sorted(bad_tiers)}")
    bad_employment = set(df["employment_type"].unique()) - VALID_EMPLOYMENT_TYPES
    if bad_employment:
        violations.append(f"customers: unexpected employment_type values {sorted(bad_employment)}")

    if (df["mean_order_value_usd"] <= 0).any():
        violations.append("customers: mean_order_value_usd has non-positive values")
    return violations


def validate_orders(df: pd.DataFrame) -> list[str]:
    violations = []
    required = ["order_id", "customer_id", "order_date", "order_value_usd",
                "months_since_acquisition", "order_month_index", "fee_revenue_usd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"orders: missing required column '{c}'" for c in missing]

    if df["order_id"].isna().any():
        violations.append("orders: order_id has nulls")
    if df["order_id"].duplicated().any():
        violations.append("orders: order_id has duplicates")
    if (df["order_value_usd"] <= 0).any():
        violations.append("orders: order_value_usd has non-positive values")
    if (df["fee_revenue_usd"] <= 0).any():
        violations.append("orders: fee_revenue_usd has non-positive values")
    if (df["months_since_acquisition"] < 0).any():
        violations.append("orders: months_since_acquisition has negative values")
    if not df["order_month_index"].between(1, N_MONTHS).all():
        violations.append(f"orders: order_month_index outside [1, {N_MONTHS}]")
    # fee_revenue_usd is a take-rate fraction of order_value_usd, so it
    # should never reach or exceed the order value it's derived from.
    if (df["fee_revenue_usd"] >= df["order_value_usd"]).any():
        violations.append("orders: fee_revenue_usd >= order_value_usd for some rows (take rate should be < 1)")
    return violations


def validate_referential_integrity(customers: pd.DataFrame, orders: pd.DataFrame) -> list[str]:
    violations = []
    if "customer_id" not in customers.columns or "customer_id" not in orders.columns:
        return violations
    orphan_ids = set(orders["customer_id"]) - set(customers["customer_id"])
    if orphan_ids:
        sample = sorted(orphan_ids)[:5]
        violations.append(f"orders: {len(orphan_ids)} customer_id values not present in customers (e.g. {sample})")
    return violations


def run_all_contracts(customers: pd.DataFrame, orders: pd.DataFrame) -> None:
    violations = (
        validate_customers(customers)
        + validate_orders(orders)
        + validate_referential_integrity(customers, orders)
    )
    if violations:
        raise DataContractError("Data contract violations:\n" + "\n".join(f"  - {v}" for v in violations))
