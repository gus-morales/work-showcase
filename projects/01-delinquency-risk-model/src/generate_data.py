"""
Synthetic data generator for a BNPL (Buy Now, Pay Later) delinquency dataset.

This mimics the shape of a real lending book at a BNPL fintech
(loan-level records with customer demographics, credit-bureau-style
features, loan terms, and a 30+ days-past-due delinquency outcome).

All data is synthetic. Relationships between features and the target are
hand-specified below (with noise) to resemble plausible credit-risk drivers,
not fit from any real portfolio.

Run:
    python src/generate_data.py
Writes:
    data/loans.csv          (~40k loans, months 1-24)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_CUSTOMERS = 12_000
N_MONTHS = 24
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

rng = np.random.default_rng(SEED)

# No real geography: just three generic metro-size tiers, since all that
# actually matters for the income model is cost-of-living tier, not any
# specific city.
CITIES = {"Tier 1 metro": "tier1", "Tier 2 metro": "tier2", "Tier 3 metro": "tier3"}
CITY_NAMES = list(CITIES.keys())
CITY_WEIGHTS = np.array([0.38, 0.32, 0.30])
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

EMPLOYMENT_TYPES = ["salaried", "self_employed", "gig_economy", "informal"]
EMPLOYMENT_WEIGHTS = [0.45, 0.20, 0.20, 0.15]

CHANNELS = ["organic", "paid_social", "partner_store", "referral"]
CHANNEL_WEIGHTS = [0.30, 0.25, 0.35, 0.10]

MERCHANT_CATS = ["electronics", "fashion", "home_goods", "travel", "education", "groceries"]
MERCHANT_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.08, 0.12]

INSTALLMENT_OPTIONS = [3, 6, 9, 12]
INSTALLMENT_WEIGHTS = [0.35, 0.35, 0.20, 0.10]


def make_customers(n):
    age = np.clip(rng.gamma(shape=9, scale=3.6, size=n) + 18, 18, 70).round().astype(int)
    city = rng.choice(CITY_NAMES, size=n, p=CITY_WEIGHTS)
    city_tier = np.array([CITIES[c] for c in city])

    employment = rng.choice(EMPLOYMENT_TYPES, size=n, p=EMPLOYMENT_WEIGHTS)
    tier_income_mult = np.select(
        [city_tier == "tier1", city_tier == "tier2", city_tier == "tier3"],
        [1.25, 1.0, 0.82],
    )
    emp_income_mult = np.select(
        [employment == "salaried", employment == "self_employed",
         employment == "gig_economy", employment == "informal"],
        [1.15, 1.05, 0.85, 0.70],
    )
    base_income = rng.lognormal(mean=9.0, sigma=0.45, size=n)  # USD monthly
    monthly_income = base_income * tier_income_mult * emp_income_mult

    tenure_months = rng.integers(1, 48, size=n)
    num_previous_loans = np.clip(rng.poisson(lam=tenure_months / 6.0), 0, 40)

    # Thin-file / bureau-style score, loosely tied to income + employment stability
    bureau_base = 300 + 400 * (
        0.35 * (np.log(monthly_income) - 8) / 2.0
        + 0.25 * (employment == "salaried").astype(float)
        + 0.15 * np.clip(num_previous_loans, 0, 10) / 10
        + 0.25 * rng.normal(0.5, 0.2, size=n)
    )
    credit_bureau_score = np.clip(bureau_base + rng.normal(0, 40, size=n), 300, 850).round().astype(int)

    prior_delay = np.clip(
        rng.exponential(scale=np.select(
            [employment == "salaried", employment == "self_employed",
             employment == "gig_economy", employment == "informal"],
            [3.0, 5.0, 8.0, 10.0]
        ), size=n) - 1,
        0, 90
    )

    active_loans_elsewhere = rng.poisson(lam=1.2, size=n)
    device = rng.choice(["android", "ios", "web"], size=n, p=[0.62, 0.23, 0.15])
    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "age": age,
        "city": city,
        "city_tier": city_tier,
        "employment_type": employment,
        "monthly_income_usd": monthly_income.round(2),
        "tenure_months_platform": tenure_months,
        "num_previous_loans": num_previous_loans,
        "credit_bureau_score": credit_bureau_score,
        "avg_prior_repayment_delay_days": prior_delay.round(1),
        "num_active_loans_elsewhere": active_loans_elsewhere,
        "device_type": device,
        "acquisition_channel": channel,
    })


def make_loans(customers, n_months=N_MONTHS):
    rows = []
    loan_id = 1
    for _, c in customers.iterrows():
        # number of loans this customer takes across the window, tenure/history dependent
        expected_loans = 1 + c["num_previous_loans"] / 8.0
        n_loans = rng.poisson(lam=min(expected_loans, 6))
        n_loans = max(n_loans, 0)
        if n_loans == 0:
            continue
        origination_months = rng.choice(np.arange(1, n_months + 1), size=n_loans, replace=True)
        for om in origination_months:
            merchant = rng.choice(MERCHANT_CATS, p=MERCHANT_WEIGHTS)
            installments = int(rng.choice(INSTALLMENT_OPTIONS, p=INSTALLMENT_WEIGHTS))
            amount = np.clip(rng.lognormal(mean=7.3, sigma=0.55), 300, 25000)
            down_payment_ratio = np.clip(rng.beta(2, 6), 0, 0.5)
            rows.append((loan_id, c["customer_id"], om, merchant, installments,
                         round(float(amount), 2), round(float(down_payment_ratio), 3)))
            loan_id += 1
    loans = pd.DataFrame(rows, columns=[
        "loan_id", "customer_id", "origination_month", "merchant_category",
        "num_installments", "loan_amount_usd", "down_payment_ratio",
    ])
    return loans


def assign_delinquency(loans, customers, n_months=N_MONTHS):
    df = loans.merge(customers, on="customer_id", how="left")

    # Late-window macro shock: inflation/rate spike in the last 3 months of the
    # window pushes delinquency up and shifts income/bureau distributions a bit,
    # used later to demonstrate drift detection.
    late_window = df["origination_month"] >= (n_months - 2)

    z = (
        -0.010 * (df["credit_bureau_score"] - 650)
        + 0.045 * df["avg_prior_repayment_delay_days"]
        + 0.35 * df["num_active_loans_elsewhere"]
        - 0.00006 * df["monthly_income_usd"]
        + 0.00035 * df["loan_amount_usd"]
        + 0.10 * df["num_installments"]
        - 0.35 * df["down_payment_ratio"] * 10
        - 0.020 * df["tenure_months_platform"]
        + np.select(
            [df["employment_type"] == "salaried", df["employment_type"] == "self_employed",
             df["employment_type"] == "gig_economy", df["employment_type"] == "informal"],
            [-0.25, 0.0, 0.35, 0.55],
        )
        + np.select(
            [df["merchant_category"] == "electronics", df["merchant_category"] == "travel"],
            [0.20, 0.15], default=0.0,
        )
        + 0.55 * late_window.astype(float)          # macro shock
        - 4.35                                         # intercept -> ~10-12% base rate
        + rng.normal(0, 0.85, size=len(df))            # idiosyncratic noise
    )
    prob = 1 / (1 + np.exp(-z))
    delinquent = rng.binomial(1, prob)
    df["delinquent_30dpd"] = delinquent
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = make_customers(N_CUSTOMERS)
    loans = make_loans(customers)
    full = assign_delinquency(loans, customers)
    full = full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out_path = OUT_DIR / "loans.csv"
    full.to_csv(out_path, index=False)
    print(f"Wrote {len(full):,} loans -> {out_path}")
    print(f"Overall delinquency rate: {full['delinquent_30dpd'].mean():.3%}")
    print(f"Delinquency rate, last 3 months (shock window): "
          f"{full.loc[full.origination_month >= N_MONTHS - 2, 'delinquent_30dpd'].mean():.3%}")
    print(f"Delinquency rate, months 1-{N_MONTHS - 3}: "
          f"{full.loc[full.origination_month < N_MONTHS - 2, 'delinquent_30dpd'].mean():.3%}")


if __name__ == "__main__":
    main()
