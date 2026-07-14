"""Synthetic source data for the two domains this project runs the same
pipeline contracts against. Each domain gets a base population (one row
per identifier, exposing the raw field behind its target(s) and a
decision-time cutoff) plus one event-level source table with rows
dated both before and after that cutoff on purpose: the Feature stage
must filter to `event_date <= cutoff` per identifier, or it leaks
post-decision information into the model, exactly the point-in-time
problem `references/best_practices.md` in the real framework this
project is modeled on warns about.

Domain A, churn: a subscription business deciding, at each customer's
90-day mark, whether they're at risk of churning in the next 30 days.
Domain B, ticket triage: a support team deciding, within 2 hours of a
ticket opening, whether it will escalate and how long it will take to
resolve, one classification target and one regression target from the
same base population, to prove the same Model Scope handles
`task_type: both`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RNG_SEED = 9

REGIONS = ["north", "south", "east", "west"]
PLAN_TIERS = ["basic", "pro", "enterprise"]
CATEGORIES = ["billing", "technical", "account", "general"]
CHANNELS = ["email", "chat", "phone"]
CUSTOMER_TIERS = ["basic", "pro", "enterprise"]


def generate_churn_domain(n_customers: int = 3000, seed: int = RNG_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    signup_date = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n_customers), unit="D")
    tenure_at_decision = rng.integers(85, 96, n_customers)  # decided around the 90-day mark
    decision_date = signup_date + pd.to_timedelta(tenure_at_decision, unit="D")
    plan_tier = rng.choice(PLAN_TIERS, n_customers, p=[0.55, 0.35, 0.10])
    region = rng.choice(REGIONS, n_customers)

    customers = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(n_customers)],
        "signup_date": signup_date,
        "decision_date": decision_date,
        "plan_tier": plan_tier,
        "region": region,
    })

    # Event-level usage, spanning both sides of each customer's decision
    # date on purpose: some rows are legitimate pre-decision signal,
    # others are post-decision and must be filtered out at feature time.
    events = []
    for _, row in customers.iterrows():
        n_pre = rng.integers(15, 45)
        n_post = rng.integers(5, 20)
        pre_dates = row["decision_date"] - pd.to_timedelta(rng.integers(1, 90, n_pre), unit="D")
        post_dates = row["decision_date"] + pd.to_timedelta(rng.integers(1, 60, n_post), unit="D")
        for event_date in np.concatenate([pre_dates.values, post_dates.values]):
            events.append((row["customer_id"], pd.Timestamp(event_date)))

    event_dates = pd.DataFrame(events, columns=["customer_id", "event_date"])
    n_events = len(event_dates)
    event_dates["logins"] = rng.poisson(2.2, n_events)
    event_dates["feature_usage_score"] = np.clip(rng.normal(55, 18, n_events), 0, 100).round(1)
    event_dates["support_tickets_opened"] = rng.poisson(0.15, n_events)
    usage_events = event_dates.sort_values(["customer_id", "event_date"]).reset_index(drop=True)

    # Risk driven only by PRE-decision behavior, so a model trained on
    # correctly-filtered features has a real, learnable signal.
    pre = usage_events[usage_events["event_date"] <= usage_events["customer_id"].map(customers.set_index("customer_id")["decision_date"])]
    agg = pre.groupby("customer_id").agg(
        avg_logins=("logins", "mean"),
        avg_usage_score=("feature_usage_score", "mean"),
        tickets_opened=("support_tickets_opened", "sum"),
    ).reindex(customers["customer_id"]).fillna(0).reset_index(drop=True)

    tier_risk = pd.Series(plan_tier).map({"basic": 0.6, "pro": 0.0, "enterprise": -0.7}).values
    logit = (
        -2.3
        + 1.7 * (agg["avg_logins"] < 1.2).astype(float)
        + 0.075 * (60 - agg["avg_usage_score"])
        + 0.15 * agg["tickets_opened"]
        + tier_risk
        + rng.normal(0, 0.35, n_customers)
    )
    prob = 1 / (1 + np.exp(-logit))
    customers["churned_next_30d"] = (rng.uniform(0, 1, n_customers) < prob).astype(int)

    return {"customers": customers, "usage_events": usage_events}


def generate_ticket_triage_domain(n_tickets: int = 2500, seed: int = RNG_SEED + 1) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    opened_at = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 200, n_tickets) * 24 * 60, unit="m") \
        + pd.to_timedelta(rng.integers(0, 24 * 60, n_tickets), unit="m")
    triage_cutoff = opened_at + pd.Timedelta(hours=2)
    category = rng.choice(CATEGORIES, n_tickets, p=[0.30, 0.35, 0.20, 0.15])
    channel = rng.choice(CHANNELS, n_tickets, p=[0.45, 0.40, 0.15])
    customer_tier = rng.choice(CUSTOMER_TIERS, n_tickets, p=[0.50, 0.35, 0.15])

    tickets = pd.DataFrame({
        "ticket_id": [f"TKT-{i:06d}" for i in range(n_tickets)],
        "customer_id": [f"CUST-{rng.integers(0, n_tickets // 2):06d}" for _ in range(n_tickets)],
        "opened_at": opened_at,
        "triage_cutoff": triage_cutoff,
        "category": category,
        "channel": channel,
        "customer_tier": customer_tier,
    })

    # Event-level agent activity, spanning the ticket's full lifecycle:
    # rows inside the 2-hour triage window are legitimate signal
    # (urgency, early reassignment); rows after it describe what
    # actually happened and must be filtered out at feature time.
    activity = []
    for _, row in tickets.iterrows():
        n_pre = rng.integers(0, 4)
        n_post = rng.integers(1, 8)
        pre_offsets = rng.integers(0, 120, n_pre)  # minutes, inside the 2h window
        post_offsets = rng.integers(121, 4000, n_post)
        for offset in np.concatenate([pre_offsets, post_offsets]):
            activity.append((row["ticket_id"], row["opened_at"] + pd.Timedelta(minutes=int(offset))))

    activity_log = pd.DataFrame(activity, columns=["ticket_id", "activity_date"])
    n_activity = len(activity_log)
    activity_log["notes_added"] = rng.poisson(1.1, n_activity)
    activity_log["reassigned"] = (rng.uniform(0, 1, n_activity) < 0.08).astype(int)
    activity_log = activity_log.sort_values(["ticket_id", "activity_date"]).reset_index(drop=True)

    pre = activity_log[activity_log["activity_date"] <= activity_log["ticket_id"].map(tickets.set_index("ticket_id")["triage_cutoff"])]
    agg = pre.groupby("ticket_id").agg(
        early_notes=("notes_added", "sum"),
        early_reassignments=("reassigned", "sum"),
    ).reindex(tickets["ticket_id"]).fillna(0).reset_index(drop=True)

    category_severity = pd.Series(category).map({"billing": 0.3, "technical": 0.55, "account": 0.15, "general": -0.2}).values
    channel_severity = pd.Series(channel).map({"phone": 0.35, "chat": 0.0, "email": -0.15}).values

    escalate_logit = (
        -1.1
        + category_severity
        + channel_severity
        + 0.5 * agg["early_reassignments"]
        + 0.06 * agg["early_notes"]
        + rng.normal(0, 0.5, n_tickets)
    )
    escalate_prob = 1 / (1 + np.exp(-escalate_logit))
    will_escalate = (rng.uniform(0, 1, n_tickets) < escalate_prob).astype(int)

    base_hours = {"billing": 4, "technical": 9, "account": 3, "general": 2}
    resolution_hours = (
        pd.Series(category).map(base_hours).values
        + 6.0 * will_escalate
        + 1.5 * agg["early_reassignments"]
        + rng.gamma(2.0, 1.4, n_tickets)
    )

    tickets["will_escalate"] = will_escalate
    tickets["resolution_hours"] = resolution_hours.round(2)

    return {"tickets": tickets, "agent_activity": activity_log}


def main():
    churn = generate_churn_domain()
    ticket = generate_ticket_triage_domain()

    churn_dir = BASE / "data" / "churn"
    ticket_dir = BASE / "data" / "ticket_triage"
    churn_dir.mkdir(parents=True, exist_ok=True)
    ticket_dir.mkdir(parents=True, exist_ok=True)

    churn["customers"].to_csv(churn_dir / "customers.csv", index=False)
    churn["usage_events"].to_csv(churn_dir / "usage_events.csv", index=False)
    ticket["tickets"].to_csv(ticket_dir / "tickets.csv", index=False)
    ticket["agent_activity"].to_csv(ticket_dir / "agent_activity.csv", index=False)

    print(f"churn: {len(churn['customers']):,} customers, {len(churn['usage_events']):,} usage events, "
          f"churn rate {churn['customers']['churned_next_30d'].mean():.1%} -> {churn_dir}")
    print(f"ticket_triage: {len(ticket['tickets']):,} tickets, {len(ticket['agent_activity']):,} activity rows, "
          f"escalation rate {ticket['tickets']['will_escalate'].mean():.1%}, "
          f"median resolution {ticket['tickets']['resolution_hours'].median():.1f}h -> {ticket_dir}")


if __name__ == "__main__":
    main()
