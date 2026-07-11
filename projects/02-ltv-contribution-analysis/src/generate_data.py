"""
Synthetic order-level transaction data for a BNPL fintech, generated so it
genuinely follows the BG/NBD + Gamma-Gamma generative process (heterogeneous
purchase rate, per-transaction dropout probability, and a customer-level
average order value with per-order noise). This is the same fictional
company as project 01 (Mexican BNPL), viewed from the growth/LTV side
instead of the credit-risk side.

Not using lifetimes.generate_data directly (it calls a pandas API removed in
pandas 2.x); this reimplements the same BG/NBD sampling logic with numpy so
the data is still exactly the process BetaGeoFitter/GammaGammaFitter expect.

Run:
    python src/generate_data.py
Writes:
    data/orders.csv     (order-level transactions)
    data/customers.csv  (customer attributes + acquisition cohort)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 7
N_CUSTOMERS = 15_000
N_MONTHS = 24
WEEKS_PER_MONTH = 4.0
TOTAL_WEEKS = N_MONTHS * WEEKS_PER_MONTH
OBS_END = pd.Timestamp("2026-01-01")

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
rng = np.random.default_rng(SEED)

# --- customer attributes (same categories/weights as project 01, for
# narrative continuity - this is "the same company, different lens") ---
CITIES = {
    "CDMX": "tier1", "Guadalajara": "tier1", "Monterrey": "tier1",
    "Puebla": "tier2", "Queretaro": "tier2", "Merida": "tier2",
    "Leon": "tier2", "Tijuana": "tier2",
    "Oaxaca": "tier3", "Chetumal": "tier3", "Zacatecas": "tier3", "Tepic": "tier3",
}
CITY_NAMES = list(CITIES.keys())
CITY_WEIGHTS = np.array([0.18, 0.10, 0.10, 0.07, 0.06, 0.06, 0.06, 0.07, 0.08, 0.08, 0.07, 0.07])
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

EMPLOYMENT_TYPES = ["salaried", "self_employed", "gig_economy", "informal"]
EMPLOYMENT_WEIGHTS = [0.45, 0.20, 0.20, 0.15]

# Channel mix shifts over the acquisition window: early cohorts skew
# organic/referral (higher quality, cheaper), later cohorts skew paid_social
# as the company scales up marketing spend (lower quality, more expensive).
# This mix shift is the hook for the contribution-analysis story.
CHANNEL_MIX_BY_STAGE = {
    "early": {"organic": 0.45, "referral": 0.20, "partner_store": 0.20, "paid_social": 0.15},
    "mid":   {"organic": 0.30, "referral": 0.15, "partner_store": 0.25, "paid_social": 0.30},
    "late":  {"organic": 0.20, "referral": 0.10, "partner_store": 0.25, "paid_social": 0.45},
}

# BG/NBD parameters per channel (weekly frequency). r/alpha = mean purchase
# rate per week (Gamma-distributed across customers). a/b = dropout
# probability after each transaction (Beta-distributed across customers).
CHANNEL_PARAMS = {
    "organic":       dict(r=1.00, alpha=4.00, a=1.0, b=9.0),   # mean 0.25/wk, mean p=0.10
    "partner_store": dict(r=1.30, alpha=3.80, a=0.8, b=10.0),  # mean 0.342/wk, mean p=0.074 (best segment)
    "referral":      dict(r=1.10, alpha=4.00, a=0.9, b=9.5),   # mean 0.275/wk, mean p=0.086
    "paid_social":   dict(r=0.70, alpha=4.50, a=1.8, b=6.0),   # mean 0.156/wk, mean p=0.231 (worst segment)
}

# Base average order value (MXN) by city tier x employment type multiplier,
# same spirit as project 01's income model.
TIER_VALUE_MULT = {"tier1": 1.25, "tier2": 1.0, "tier3": 0.82}
EMPLOYMENT_VALUE_MULT = {"salaried": 1.15, "self_employed": 1.05, "gig_economy": 0.85, "informal": 0.70}
BASE_ORDER_VALUE = 550.0  # MXN
TAKE_RATE = 0.065  # merchant/interest fee revenue as a share of order value


def assign_cohort_month(n):
    # growing acquisition volume over time (typical early-stage fintech growth)
    weights = np.linspace(1, 3, N_MONTHS)
    weights = weights / weights.sum()
    return rng.choice(np.arange(1, N_MONTHS + 1), size=n, p=weights)


def stage_for_month(m):
    if m <= 8:
        return "early"
    if m <= 16:
        return "mid"
    return "late"


def assign_channel(cohort_months):
    channels = np.empty(len(cohort_months), dtype=object)
    for stage in ("early", "mid", "late"):
        mask = np.array([stage_for_month(m) == stage for m in cohort_months])
        n = mask.sum()
        if n == 0:
            continue
        mix = CHANNEL_MIX_BY_STAGE[stage]
        channels[mask] = rng.choice(list(mix.keys()), size=n, p=list(mix.values()))
    return channels


def make_customers(n):
    cohort_month = assign_cohort_month(n)
    channel = assign_channel(cohort_month)
    city = rng.choice(CITY_NAMES, size=n, p=CITY_WEIGHTS)
    city_tier = np.array([CITIES[c] for c in city])
    employment = rng.choice(EMPLOYMENT_TYPES, size=n, p=EMPLOYMENT_WEIGHTS)

    mult = np.array([TIER_VALUE_MULT[t] for t in city_tier]) * \
           np.array([EMPLOYMENT_VALUE_MULT[e] for e in employment])
    # customer-level mean order value (Gamma-Gamma: avg value varies by customer)
    mean_order_value = rng.gamma(shape=6.0, scale=(BASE_ORDER_VALUE * mult) / 6.0)

    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "cohort_month": cohort_month,
        "acquisition_channel": channel,
        "city": city,
        "city_tier": city_tier,
        "employment_type": employment,
        "mean_order_value_mxn": mean_order_value.round(2),
    })


def simulate_transactions(customers):
    """BG/NBD-consistent simulation: per-customer Poisson purchase process
    with heterogeneous rate (Gamma) and post-purchase dropout probability
    (Beta), simulated directly with numpy."""
    rows = []
    for channel, params in CHANNEL_PARAMS.items():
        seg = customers[customers.acquisition_channel == channel]
        if len(seg) == 0:
            continue
        lam = rng.gamma(shape=params["r"], scale=1.0 / params["alpha"], size=len(seg))
        p_dropout = rng.beta(params["a"], params["b"], size=len(seg))
        obs_weeks = (N_MONTHS - seg["cohort_month"].values + 1) * WEEKS_PER_MONTH

        for (idx, cust), lam_i, p_i, T_i in zip(seg.iterrows(), lam, p_dropout, obs_weeks):
            t = 0.0
            alive = True
            # first purchase happens at acquisition (t=0) - this is the
            # signup order, always present
            week_offsets = [0.0]
            while alive:
                wait = rng.exponential(scale=1.0 / lam_i)
                t += wait
                if t >= T_i:
                    break
                week_offsets.append(t)
                if rng.random() < p_i:
                    alive = False

            cohort_start = OBS_END - pd.Timedelta(weeks=(N_MONTHS - cust["cohort_month"] + 1) * WEEKS_PER_MONTH)
            for wk in week_offsets:
                order_date = cohort_start + pd.Timedelta(weeks=wk)
                order_value = rng.gamma(shape=6.0, scale=cust["mean_order_value_mxn"] / 6.0)
                months_since_acq = int(wk // WEEKS_PER_MONTH)
                order_month_index = min(int(cust["cohort_month"]) + months_since_acq, N_MONTHS)
                rows.append((cust["customer_id"], order_date, round(float(order_value), 2),
                             months_since_acq, order_month_index))

    orders = pd.DataFrame(rows, columns=[
        "customer_id", "order_date", "order_value_mxn", "months_since_acquisition", "order_month_index",
    ])
    orders = orders.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    orders.insert(0, "order_id", np.arange(1, len(orders) + 1))
    orders["fee_revenue_mxn"] = (orders["order_value_mxn"] * TAKE_RATE).round(2)
    orders["order_month"] = orders["order_date"].dt.to_period("M").astype(str)
    return orders


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customers = make_customers(N_CUSTOMERS)
    orders = simulate_transactions(customers)

    customers.to_csv(OUT_DIR / "customers.csv", index=False)
    orders.to_csv(OUT_DIR / "orders.csv", index=False)

    print(f"Wrote {len(customers):,} customers -> data/customers.csv")
    print(f"Wrote {len(orders):,} orders -> data/orders.csv")
    print(f"Orders per customer: mean={orders.groupby('customer_id').size().mean():.2f}, "
          f"median={orders.groupby('customer_id').size().median():.0f}")
    print(f"Channel mix:\n{customers['acquisition_channel'].value_counts(normalize=True).round(3)}")
    by_channel_orders = orders.merge(customers[["customer_id", "acquisition_channel"]], on="customer_id")
    print(f"Orders/customer by channel:\n"
          f"{by_channel_orders.groupby('acquisition_channel').size() / customers.groupby('acquisition_channel').size()}")


if __name__ == "__main__":
    main()
