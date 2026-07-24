"""
Synthetic data generator for a cardholder growth dataset: a fictional
regional bank's card business, viewed from the growth/marketing team's
side. One row per customer, covering their trailing
90-day behavior (recency, order frequency, spend, category mix,
checkout authorization decline rate) plus whether the growth team's
past win-back campaign reached them and whether they responded.

Two latent traits drive the data and are deliberately generated
independent of each other, then only observed through noisy proxies:

- `baseline_value` (never written to the output CSV): a customer's
  long-run spending potential. Response to a growth offer is driven by
  this, not by how active the customer happens to be right now.
- `current_engagement`: how active the customer is in the trailing
  90-day window specifically. Generated independent of
  `baseline_value`, so a high-potential customer can show up currently
  dormant, and a currently active customer isn't necessarily high
  long-run potential. This is what makes "dormant" a genuinely mixed
  segment behaviorally, some dormant customers are worth winning back,
  most aren't, and segmentation on current-window behavior alone can't
  tell them apart.

`decline_rate` is included as a feature but has zero true weight in
the response-generating formula below, a decoy covariate: a
well-behaved propensity model should learn to downweight it.

Run:
    python src/generate_data.py
Writes:
    data/customers.csv (~12k customers)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_CUSTOMERS = 12_000
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

CHANNELS = ["android", "ios", "web"]
CHANNEL_WEIGHTS = [0.62, 0.23, 0.15]

rng = np.random.default_rng(SEED)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_customers(n=N_CUSTOMERS):
    customer_id = np.arange(1, n + 1)
    tenure_days = np.clip(rng.exponential(scale=280, size=n), 30, 1500).round(0).astype(int)

    # Latent long-run spending potential. Kept out of the output CSV:
    # a real growth team wouldn't have direct access to it either, only
    # to behavioral proxies like tenure and order history.
    baseline_value = rng.gamma(shape=3.0, scale=1.0, size=n)

    # Cumulative order count since signup, a noisy proxy for baseline
    # value that IS observable (it's just a running counter), unlike
    # baseline_value itself.
    lifetime_lambda = np.clip(tenure_days / 25 * (0.4 + 0.4 * baseline_value), 0.5, None)
    lifetime_orders = rng.poisson(lifetime_lambda) + 1

    # Current-window engagement, generated independent of baseline
    # value on purpose (see module docstring): whether a customer
    # happens to be active right now doesn't imply anything about how
    # valuable they've historically been.
    current_engagement = rng.beta(2.0, 2.0, size=n)

    return pd.DataFrame({
        "customer_id": customer_id,
        "tenure_days": tenure_days,
        "_baseline_value": baseline_value,  # latent, dropped before saving
        "lifetime_orders": lifetime_orders,
        "_current_engagement": current_engagement,  # latent, dropped before saving
    })


def add_behavioral_features(df):
    n = len(df)
    engagement = df["_current_engagement"].values
    baseline_value = df["_baseline_value"].values

    recency_days = np.clip(180 * (1 - engagement) + rng.normal(0, 15, n), 0, 180).round(0).astype(int)
    frequency_90d = rng.poisson(np.clip(10 * engagement, 0.1, None))

    avg_ticket = 15 + 12 * baseline_value + rng.lognormal(mean=0.0, sigma=0.25, size=n)
    monetary_90d = np.clip(frequency_90d * avg_ticket * rng.lognormal(mean=0.0, sigma=0.3, size=n), 0, None).round(2)

    category_diversity = np.clip(
        np.round(1 + 4 * engagement + rng.normal(0, 0.6, n)), 1, 6,
    ).astype(int)
    category_diversity = np.minimum(category_diversity, np.maximum(frequency_90d, 1))

    # Checkout authorization decline rate: higher for thinner-file
    # (lower-tenure) accounts and, mildly, for higher-velocity
    # customers, a realistic risk-driven decline pattern. Has zero
    # weight in the response formula below, a decoy feature.
    decline_rate = np.clip(
        0.22 - 0.00012 * df["tenure_days"].values + 0.010 * frequency_90d + rng.normal(0, 0.05, n),
        0.01, 0.60,
    ).round(3)

    channel = rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS)

    df = df.copy()
    df["recency_days"] = recency_days
    df["frequency_90d"] = frequency_90d
    df["monetary_90d"] = monetary_90d
    df["category_diversity"] = category_diversity
    df["decline_rate"] = decline_rate
    df["primary_channel"] = channel
    return df


def assign_offer_and_response(df):
    """The growth team's past win-back campaign wasn't randomized: it
    leaned toward moderately lapsed, established accounts (a standard
    win-back targeting rule), plus a broad baseline of other outreach.
    Response is driven by baseline_value and a "sweet spot" recency
    band (moderately lapsed responds best, not currently-active and
    not fully dormant), independent of decline_rate."""
    n = len(df)
    recency = df["recency_days"].values
    tenure = df["tenure_days"].values
    baseline_value = df["_baseline_value"].values

    lapsed_band = ((recency >= 30) & (recency <= 120)).astype(float)
    established = (tenure > 90).astype(float)
    logit_offer = -1.35 + 1.10 * lapsed_band + 0.50 * established + rng.normal(0, 0.4, n)
    offer_sent = rng.binomial(1, sigmoid(logit_offer))

    sweet_spot = -0.00028 * (recency - 70) ** 2
    z_respond = -2.35 + 0.60 * baseline_value + sweet_spot + rng.normal(0, 0.6, n)
    responded_draw = rng.binomial(1, sigmoid(z_respond))

    df = df.copy()
    df["past_offer_sent"] = offer_sent
    # `responded` is only ever observed for customers who were actually
    # offered something; NaN (not 0) for everyone else, "never given a
    # chance" is a different state from "given a chance and declined".
    df["responded"] = np.where(offer_sent == 1, responded_draw, np.nan)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = make_customers(N_CUSTOMERS)
    df = add_behavioral_features(df)
    df = assign_offer_and_response(df)

    out = df.drop(columns=["_baseline_value", "_current_engagement"])
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_path = OUT_DIR / "customers.csv"
    out.to_csv(out_path, index=False)

    offered = out[out["past_offer_sent"] == 1]
    print(f"Wrote {len(out):,} customers -> {out_path}")
    print(f"Past offer sent: {len(offered):,} ({len(offered) / len(out):.1%} of customers)")
    print(f"Response rate among offered: {offered['responded'].mean():.1%}")
    lapsed_offered = offered[(offered["recency_days"] >= 30) & (offered["recency_days"] <= 120)]
    active_offered = offered[offered["recency_days"] < 30]
    dormant_offered = offered[offered["recency_days"] > 120]
    print(f"Response rate, lapsed (30-120d) vs. active (<30d) vs. dormant (>120d): "
          f"{lapsed_offered['responded'].mean():.1%} vs. "
          f"{active_offered['responded'].mean():.1%} vs. "
          f"{dormant_offered['responded'].mean():.1%}")


if __name__ == "__main__":
    main()
