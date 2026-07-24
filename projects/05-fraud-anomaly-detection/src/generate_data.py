"""
Synthetic data generator for a bank's card-transaction fraud dataset: a
fictional regional bank's card business, viewed from the fraud/risk
operations side. Transaction-level records with device, velocity, and
mismatch signals available at checkout time, and a fraud outcome
(`is_fraud`) at a realistic, heavily imbalanced rate.

All data is synthetic. Relationships between features and the fraud
outcome are hand-specified below (with noise) to resemble plausible
fraud-risk drivers (new/unrecognized device, billing/shipping and
IP/billing country mismatches, checkout velocity, account age, checkout
speed), not fit from any real portfolio. Features are generated first,
independent of the fraud outcome, and the label is computed from them
afterward, the same generation order used in every other project in
this repo, so that "the model recovers known relationships" in the
interpretability section is a genuine check rather than a foregone
conclusion.

Run:
    python src/generate_data.py
Writes:
    data/transactions.csv (~90k transactions, 15k customers, 180 days)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_CUSTOMERS = 15_000
N_DAYS = 180
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

rng = np.random.default_rng(SEED)

# Merchant category and device vocabularies for online card transactions.
MERCHANT_CATS = ["electronics", "fashion", "home_goods", "travel", "education", "groceries"]
MERCHANT_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.08, 0.12]
DEVICE_TYPES = ["android", "ios", "web"]
DEVICE_WEIGHTS = [0.62, 0.23, 0.15]

BURST_CUSTOMER_SHARE = 0.10  # share of customers with at least one rapid-fire burst of transactions
INTERCEPT = -3.20  # calibrated to a ~1.4% base fraud rate


def make_customers(n=N_CUSTOMERS):
    customer_id = np.arange(1, n + 1)
    account_age_at_start_days = np.clip(rng.exponential(scale=250, size=n), 0, 1500).round(0)
    home_device = rng.choice(DEVICE_TYPES, size=n, p=DEVICE_WEIGHTS)
    # Customer-level baseline spend level, drives each customer's typical
    # transaction amount before per-transaction noise.
    spend_scale = rng.gamma(shape=3.0, scale=1.0, size=n)
    n_transactions = np.clip(rng.poisson(lam=6, size=n), 1, 60)

    return pd.DataFrame({
        "customer_id": customer_id,
        "account_age_at_start_days": account_age_at_start_days.astype(int),
        "home_device": home_device,
        "spend_scale": spend_scale,
        "n_transactions": n_transactions,
    })


def make_transactions(customers):
    rows_customer_id = np.repeat(customers["customer_id"].values, customers["n_transactions"].values)
    rows_account_age = np.repeat(customers["account_age_at_start_days"].values, customers["n_transactions"].values)
    rows_home_device = np.repeat(customers["home_device"].values, customers["n_transactions"].values)
    rows_spend_scale = np.repeat(customers["spend_scale"].values, customers["n_transactions"].values)
    n_total = len(rows_customer_id)

    # Transaction day within the window: mostly spread uniformly across
    # each customer's active period. A subset of customers (independent
    # of any fraud outcome, decided purely by this flag) get 2-5 of their
    # transactions clustered into a single rapid-fire burst instead,
    # since bursts of activity are a realistic pattern on their own
    # (a shopping spree, a compromised session) that the fraud label
    # below will weight, not something manufactured to match the label.
    day_offset = rng.uniform(0, N_DAYS, size=n_total)
    is_burst_customer = rng.random(len(customers)) < BURST_CUSTOMER_SHARE
    burst_customer_ids = set(customers.loc[is_burst_customer, "customer_id"])

    df = pd.DataFrame({
        "customer_id": rows_customer_id,
        "account_age_at_start_days": rows_account_age,
        "home_device": rows_home_device,
        "spend_scale": rows_spend_scale,
        "day_offset": day_offset,
    })

    # Apply bursts: for each burst customer, pick a random burst center
    # and pull 2-5 of their transactions to within a ~40-minute window of it.
    for cid in burst_customer_ids:
        idx = df.index[df["customer_id"] == cid]
        if len(idx) < 2:
            continue
        burst_size = min(len(idx), rng.integers(2, 6))
        burst_idx = rng.choice(idx, size=burst_size, replace=False)
        center_day = rng.uniform(0, N_DAYS)
        jitter_days = rng.uniform(-20, 20, size=burst_size) / (24 * 60)  # +/- 20 minutes
        df.loc[burst_idx, "day_offset"] = np.clip(center_day + jitter_days, 0, N_DAYS - 0.001)

    df = df.sort_values(["customer_id", "day_offset"]).reset_index(drop=True)
    df["transaction_id"] = np.arange(1, len(df) + 1)

    epoch = pd.Timestamp("2025-01-01")
    df["timestamp"] = epoch + pd.to_timedelta(df["day_offset"], unit="D")
    df["account_age_days_at_tx"] = (df["account_age_at_start_days"] + df["day_offset"]).round(0).astype(int)

    df["merchant_category"] = rng.choice(MERCHANT_CATS, size=len(df), p=MERCHANT_WEIGHTS)
    # Device usually matches the customer's home device; a different
    # device on a transaction is exactly what "new/unrecognized device"
    # means operationally, so it's derived rather than drawn separately.
    other_device_roll = rng.random(len(df))
    other_device_choice = rng.choice(DEVICE_TYPES, size=len(df))
    device_type = np.where(other_device_roll < 0.08, other_device_choice, df["home_device"].values)
    df["device_type"] = device_type
    df["is_new_device"] = (df["device_type"] != df["home_device"]).astype(int)

    df["billing_shipping_mismatch"] = rng.binomial(1, 0.05, size=len(df))
    df["ip_billing_country_mismatch"] = rng.binomial(1, 0.03, size=len(df))

    amount_noise = rng.lognormal(mean=0.0, sigma=0.55, size=len(df))
    df["amount_usd"] = np.clip(df["spend_scale"] * 18 * amount_noise, 3, 6000).round(2)

    # Checkout duration: most transactions take tens of seconds; a
    # lognormal's left tail naturally produces some very fast ones
    # (saved payment info, or a scripted checkout) without needing to
    # condition on the fraud outcome.
    df["checkout_seconds"] = np.clip(rng.lognormal(mean=np.log(38), sigma=0.5, size=len(df)), 2, 400).round(1)

    return df.drop(columns=["day_offset", "account_age_at_start_days", "home_device", "spend_scale"])


def add_velocity_features(df):
    """Rolling transaction counts per customer in the 1-hour and 24-hour
    windows before each transaction (excluding the transaction itself),
    the standard checkout-time velocity signal fraud systems use.
    Computed with pandas' time-aware rolling window, grouped by customer,
    which is vectorized and fast even at tens of thousands of rows."""
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    indexed = df.set_index("timestamp")

    count_1h = indexed.groupby("customer_id")["transaction_id"].rolling("1h").count()
    count_24h = indexed.groupby("customer_id")["transaction_id"].rolling("24h").count()

    df["transactions_last_1h"] = (count_1h.values - 1).astype(int)
    df["transactions_last_24h"] = (count_24h.values - 1).astype(int)
    return df


def assign_fraud_label(df):
    z = (
        2.30 * df["is_new_device"]
        + 1.85 * df["billing_shipping_mismatch"]
        + 2.60 * df["ip_billing_country_mismatch"]
        + 0.38 * df["transactions_last_1h"]
        + 0.10 * df["transactions_last_24h"]
        - 0.0050 * df["account_age_days_at_tx"].clip(upper=1500)
        - 0.028 * df["checkout_seconds"]
        + 0.00035 * df["amount_usd"]
        + INTERCEPT
        + rng.normal(0, 1.0, size=len(df))
    )
    prob = 1 / (1 + np.exp(-z))
    df = df.copy()
    df["is_fraud"] = rng.binomial(1, prob)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = make_customers(N_CUSTOMERS)
    transactions = make_transactions(customers)
    transactions = add_velocity_features(transactions)
    full = assign_fraud_label(transactions)
    full = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_path = OUT_DIR / "transactions.csv"
    full.to_csv(out_path, index=False)
    print(f"Wrote {len(full):,} transactions across {full['customer_id'].nunique():,} customers -> {out_path}")
    print(f"Overall fraud rate: {full['is_fraud'].mean():.3%}")
    print(f"Fraud rate, new device vs. recognized device: "
          f"{full.loc[full.is_new_device == 1, 'is_fraud'].mean():.3%} vs. "
          f"{full.loc[full.is_new_device == 0, 'is_fraud'].mean():.3%}")
    print(f"Fraud rate, transactions_last_1h >= 2 vs. 0: "
          f"{full.loc[full.transactions_last_1h >= 2, 'is_fraud'].mean():.3%} vs. "
          f"{full.loc[full.transactions_last_1h == 0, 'is_fraud'].mean():.3%}")


if __name__ == "__main__":
    main()
