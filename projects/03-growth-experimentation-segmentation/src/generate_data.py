"""
Synthetic data for project 03, same fictional BNPL fintech as
projects 01 and 02, viewed from the growth/experimentation and lifecycle
marketing side. Four datasets, each feeding one section of the analysis:

1. experiment_users.csv   - an A/B test on a new repayment-reminder flow,
   with a pre-period revenue covariate (for CUPED) and a post-period
   binary conversion outcome + revenue outcome.
2. regional_rollout.csv   - a daily region-level panel for a phased
   feature rollout, used for a difference-in-differences analysis where
   randomization isn't available.
3. rfm_customers.csv      - customer-level transaction summaries for
   RFM segmentation.
4. support_tickets.csv    - synthetic support ticket text across five
   underlying topics, for light NLP topic modeling.

Run:
    python src/generate_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 11
rng = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------
# 1. A/B test: repayment-reminder flow
# ---------------------------------------------------------------------

def make_experiment_users(n=20_000):
    """Users randomized into control (current reminder) vs. treatment
    (redesigned reminder with a clearer due-date and one-tap repayment
    link). Outcome: whether they repaid on time and their revenue
    (fees) in the 14 days after exposure. A pre-period revenue covariate
    is included for CUPED variance reduction; it's correlated with the
    outcome but unaffected by treatment (measured before assignment)."""
    user_id = np.arange(1, n + 1)
    arm = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])

    # Latent customer quality drives both pre- and post-period behavior.
    # Kept a strong, dominant driver of both revenue windows (relative to
    # noise) since that's what makes the pre-period covariate useful for
    # CUPED; a weak covariate would make the variance-reduction step a
    # no-op, which would defeat the point of demonstrating it.
    quality = rng.gamma(shape=3.0, scale=1.0, size=n)
    revenue_pre_30d = np.clip(rng.normal(loc=quality * 28, scale=18, size=n), 0, None).round(2)

    base_conversion = 0.34 + 0.09 * (quality - quality.mean()) / quality.std()
    base_conversion = np.clip(base_conversion, 0.03, 0.95)
    treatment_lift = np.where(arm == "treatment", 0.035, 0.0)  # +3.5pp absolute
    p_convert = np.clip(base_conversion + treatment_lift, 0.01, 0.99)
    converted = rng.binomial(1, p_convert)

    noise = rng.normal(0, 20, size=n)
    revenue_post_14d = np.clip(
        converted * (quality * 22 + 15) + noise, 0, None
    ).round(2)

    return pd.DataFrame({
        "user_id": user_id,
        "arm": arm,
        "revenue_pre_30d_usd": revenue_pre_30d,
        "converted_post_14d": converted,
        "revenue_post_14d_usd": revenue_post_14d,
    })


# ---------------------------------------------------------------------
# 2. Regional rollout panel (difference-in-differences)
# ---------------------------------------------------------------------

def make_regional_rollout(n_regions=40, n_days=180, rollout_day=100):
    """A new in-app collections feature is rolled out to half the
    regions (chosen by business priority, not randomized) on
    `rollout_day`. Daily on-time repayment rate per region, with region
    fixed effects, a common time trend, and a true treatment effect
    after rollout in treated regions only."""
    region_id = np.arange(1, n_regions + 1)
    treated = (region_id <= n_regions // 2)  # first half rolled out first (non-random on purpose)
    region_base_rate = rng.normal(0.62, 0.05, size=n_regions)

    rows = []
    for i, rid in enumerate(region_id):
        is_treated = treated[i]
        base = region_base_rate[i]
        for day in range(n_days):
            common_trend = 0.00015 * day  # slow, shared secular improvement
            post = day >= rollout_day
            treatment_effect = 0.04 if (is_treated and post) else 0.0
            noise = rng.normal(0, 0.02)
            rate = np.clip(base + common_trend + treatment_effect + noise, 0.05, 0.98)
            n_customers = rng.integers(80, 260)
            on_time = rng.binomial(n_customers, rate)
            rows.append((rid, "treated" if is_treated else "control", day, post,
                         n_customers, on_time, on_time / n_customers))

    return pd.DataFrame(rows, columns=[
        "region_id", "group", "day", "post_rollout",
        "n_customers", "n_on_time", "on_time_rate",
    ])


# ---------------------------------------------------------------------
# 3. RFM customer transaction summaries
# ---------------------------------------------------------------------

def make_rfm_customers(n=8_000, obs_end=pd.Timestamp("2026-07-01"), window_days=365):
    """Independent, simpler transaction generator than project 02's
    BG/NBD one; built directly at the RFM summary level since that's
    all this analysis needs (no need for order-level granularity).

    Four latent behavior archetypes with reasonably separated recency,
    frequency, and value profiles, so the resulting customer base has
    genuine cluster structure for KMeans to recover rather than one
    smooth blob. Real RFM data is messier than this, but a portfolio
    piece demonstrating segmentation should have segments actually
    worth demonstrating."""
    customer_id = np.arange(1, n + 1)

    archetype = rng.choice(
        ["champion", "loyal", "at_risk", "dormant"], size=n, p=[0.15, 0.30, 0.25, 0.30],
    )
    freq_lambda = {"champion": 24, "loyal": 10, "at_risk": 7, "dormant": 1.5}
    recency_range = {"champion": (0, 20), "loyal": (20, 70), "at_risk": (100, 220), "dormant": (260, 400)}
    avg_order = {"champion": 720, "loyal": 600, "at_risk": 560, "dormant": 470}

    recency_days = np.empty(n)
    frequency = np.empty(n, dtype=int)
    monetary = np.empty(n)

    for arch, lam in freq_lambda.items():
        mask = archetype == arch
        cnt = mask.sum()
        frequency[mask] = rng.poisson(lam=lam, size=cnt) + 1
        lo, hi = recency_range[arch]
        recency_days[mask] = rng.uniform(lo, hi, size=cnt)
        monetary[mask] = np.clip(
            rng.gamma(shape=frequency[mask], scale=avg_order[arch] / 1.4), 200, None
        )

    recency_days = np.clip(recency_days, 0, window_days).round(0)
    last_order_date = obs_end - pd.to_timedelta(recency_days, unit="D")

    return pd.DataFrame({
        "customer_id": customer_id,
        "last_order_date": last_order_date,
        "recency_days": recency_days.astype(int),
        "frequency": frequency,
        "monetary_usd": monetary.round(2),
    })


# ---------------------------------------------------------------------
# 4. Support ticket text (light NLP topic modeling)
# ---------------------------------------------------------------------

TOPIC_TEMPLATES = {
    "late_fee_dispute": [
        "I was charged a late fee but I paid on time, please review my late fee",
        "Why do I have a late payment fee, my card was charged before the due date",
        "The late fee on my last installment seems wrong, can someone reverse this late fee",
        "I got billed twice this month plus a late fee I do not think I deserve",
        "Disputing a late payment fee, I have proof of payment before the deadline",
        "This late fee should not be on my account, my autopay went through on time",
        "Please remove the late fee, my bank shows the payment cleared before the due date",
        "I am being charged a late payment fee every month even though I pay early",
        "The late fee amount does not match my contract, please review this late charge",
        "Second time this late fee has appeared incorrectly, please fix my late payment charge",
    ],
    "app_bug": [
        "The app crashes every time I try to open my payment schedule",
        "The app freezes and I cannot see my installment plan, the app just crashes",
        "The app shows a blank screen after checkout, the app is completely broken",
        "The app keeps logging me out in the middle of a payment, very buggy app",
        "Getting an error message in the app when I try to update my card",
        "The mobile app will not load my account, the app crashes on startup",
        "Every time I open the app it crashes before I can see my balance",
        "The app is stuck loading and never finishes, please fix this app bug",
        "I keep getting an app error code whenever I try to make a payment",
        "The app crashed and now none of my payment history is showing up",
    ],
    "refund_request": [
        "I returned the item to the store and need my installment plan refunded",
        "Can you process a refund and cancel my plan, the order was cancelled",
        "Requesting a refund since the merchant already refunded me directly",
        "I want a refund for this purchase and to cancel my first payment",
        "The store processed a return but I still need a refund on my installments",
        "Please refund my down payment, I returned the item to the merchant",
        "I need a refund because the product arrived damaged and I sent it back",
        "Can I get a refund and cancellation since I no longer want this order",
        "My refund has not shown up yet even though the store confirmed the return",
        "Requesting a full refund and plan cancellation for a cancelled order",
    ],
    "kyc_verification": [
        "My identity verification has been stuck in review for a week",
        "I uploaded my ID for verification twice and still cannot get approved",
        "Verification keeps failing even though my identity documents are valid",
        "How long does identity verification usually take, mine is still pending",
        "I need help completing verification, the ID document upload keeps failing",
        "My account verification was rejected but my documents are correct",
        "Please help me finish identity verification, I have tried three times",
        "The KYC verification process will not accept my photo ID",
        "I cannot pass verification, it keeps asking for the same documents",
        "My verification status has said pending for over five days now",
    ],
    "general_inquiry": [
        "How do I increase my available credit limit for future purchases",
        "What happens if I pay off my installment plan early, is there a discount",
        "Can I change the due date of my monthly installment payments",
        "Do you offer any loyalty rewards for customers who always pay on time",
        "I want to know more about how the interest is calculated on my plan",
        "What is the maximum credit limit available for new customers",
        "Is there a way to raise my credit limit after a few on-time payments",
        "How does the rewards program work for frequent customers",
        "Can you explain how the interest rate is applied to my installments",
        "What are the eligibility requirements for a higher credit limit",
    ],
}

FILLER_PREFIXES = ["Hi, ", "Hello, ", "Hey, ", "Good afternoon, ", ""]
FILLER_SUFFIXES = [" Thank you.", " Please help.", " Appreciate any update.", " Thanks in advance.", ""]


def make_support_tickets(n=1500):
    topics = list(TOPIC_TEMPLATES.keys())
    topic_weights = [0.28, 0.22, 0.18, 0.17, 0.15]
    rows = []
    ticket_id = 1
    for _ in range(n):
        topic = rng.choice(topics, p=topic_weights)
        template = rng.choice(TOPIC_TEMPLATES[topic])
        text = rng.choice(FILLER_PREFIXES) + template + rng.choice(FILLER_SUFFIXES)
        rows.append((ticket_id, topic, text))
        ticket_id += 1
    return pd.DataFrame(rows, columns=["ticket_id", "true_topic", "ticket_text"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp = make_experiment_users()
    exp.to_csv(OUT_DIR / "experiment_users.csv", index=False)
    print(f"Wrote {len(exp):,} experiment users -> data/experiment_users.csv")
    print(exp.groupby("arm")["converted_post_14d"].mean())

    rollout = make_regional_rollout()
    rollout.to_csv(OUT_DIR / "regional_rollout.csv", index=False)
    print(f"Wrote {len(rollout):,} region-days -> data/regional_rollout.csv")

    rfm = make_rfm_customers()
    rfm.to_csv(OUT_DIR / "rfm_customers.csv", index=False)
    print(f"Wrote {len(rfm):,} customers -> data/rfm_customers.csv")

    tickets = make_support_tickets()
    tickets.to_csv(OUT_DIR / "support_tickets.csv", index=False)
    print(f"Wrote {len(tickets):,} tickets -> data/support_tickets.csv")
    print(tickets["true_topic"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
