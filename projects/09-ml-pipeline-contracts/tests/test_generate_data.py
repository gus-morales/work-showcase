import pandas as pd

from generate_data import generate_churn_domain, generate_ticket_triage_domain


def test_churn_domain_shapes_and_keys():
    data = generate_churn_domain(n_customers=200, seed=1)
    customers, events = data["customers"], data["usage_events"]

    assert len(customers) == 200
    assert customers["customer_id"].is_unique
    assert set(events["customer_id"]).issubset(set(customers["customer_id"]))
    assert customers["churned_next_30d"].isin([0, 1]).all()
    assert 0.0 < customers["churned_next_30d"].mean() < 1.0  # not degenerate


def test_churn_domain_has_events_on_both_sides_of_decision_date():
    """The point-in-time guard has nothing to prove if every event is
    conveniently pre-decision; this dataset needs real post-decision
    rows too."""
    data = generate_churn_domain(n_customers=300, seed=2)
    customers, events = data["customers"], data["usage_events"]
    merged = events.merge(customers[["customer_id", "decision_date"]], on="customer_id")
    n_pre = (merged["event_date"] <= merged["decision_date"]).sum()
    n_post = (merged["event_date"] > merged["decision_date"]).sum()
    assert n_pre > 0
    assert n_post > 0


def test_churn_domain_is_deterministic_given_seed():
    a = generate_churn_domain(n_customers=100, seed=7)["customers"]
    b = generate_churn_domain(n_customers=100, seed=7)["customers"]
    pd.testing.assert_frame_equal(a, b)


def test_ticket_triage_domain_shapes_and_keys():
    data = generate_ticket_triage_domain(n_tickets=150, seed=1)
    tickets, activity = data["tickets"], data["agent_activity"]

    assert len(tickets) == 150
    assert tickets["ticket_id"].is_unique
    assert set(activity["ticket_id"]).issubset(set(tickets["ticket_id"]))
    assert tickets["will_escalate"].isin([0, 1]).all()
    assert (tickets["resolution_hours"] > 0).all()
    assert 0.0 < tickets["will_escalate"].mean() < 1.0


def test_ticket_triage_activity_spans_the_triage_cutoff():
    data = generate_ticket_triage_domain(n_tickets=200, seed=3)
    tickets, activity = data["tickets"], data["agent_activity"]
    merged = activity.merge(tickets[["ticket_id", "triage_cutoff"]], on="ticket_id")
    n_pre = (merged["activity_date"] <= merged["triage_cutoff"]).sum()
    n_post = (merged["activity_date"] > merged["triage_cutoff"]).sum()
    assert n_pre > 0
    assert n_post > 0


def test_escalation_raises_resolution_hours():
    """Ground-truth sanity check: an escalated ticket should take longer
    to resolve on average, since escalation time is baked into
    resolution_hours by construction."""
    data = generate_ticket_triage_domain(n_tickets=1000, seed=5)
    tickets = data["tickets"]
    escalated = tickets[tickets["will_escalate"] == 1]["resolution_hours"].mean()
    not_escalated = tickets[tickets["will_escalate"] == 0]["resolution_hours"].mean()
    assert escalated > not_escalated
