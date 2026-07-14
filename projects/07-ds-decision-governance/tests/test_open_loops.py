"""Unit tests for the overdue-detection logic, on hand-built records
with known due dates, plus a check that the real example records
produce the two seeded overdue checks."""
from datetime import date
from pathlib import Path

from open_loops import find_overdue, load_records
from schema import DecisionRecord

BASE = Path(__file__).resolve().parents[1]


def _record(**monitoring_overrides):
    monitoring = {
        "ship_check": {"due": "2026-01-10", "done": True},
        "outcome_check": {"due": "2026-01-20", "done": True},
    }
    monitoring.update(monitoring_overrides)
    return DecisionRecord(
        id="DSG-TEST", title="Test", artifact_type="dashboard_change",
        domain="product_analytics", impact_level="medium", status="shipped",
        author="Tester",
        dates={"proposed": "2026-01-01", "approved": "2026-01-02", "shipped": "2026-01-05"},
        monitoring=monitoring,
        reviewers=["A", "B"],
    )


def test_overdue_and_not_done_is_flagged():
    record = _record(ship_check={"due": "2026-01-10", "done": False})
    overdue = find_overdue([(Path("x.md"), record)], as_of=date(2026, 2, 1))
    assert len(overdue) == 1
    assert overdue[0]["check"] == "ship_check"
    assert overdue[0]["days_overdue"] == 22


def test_done_check_is_not_flagged_even_if_past_due():
    record = _record(ship_check={"due": "2026-01-10", "done": True})
    overdue = find_overdue([(Path("x.md"), record)], as_of=date(2026, 2, 1))
    assert overdue == []


def test_not_yet_due_is_not_flagged():
    record = _record(ship_check={"due": "2026-03-01", "done": False})
    overdue = find_overdue([(Path("x.md"), record)], as_of=date(2026, 2, 1))
    assert overdue == []


def test_both_checks_can_be_overdue_at_once():
    record = _record(
        ship_check={"due": "2026-01-10", "done": False},
        outcome_check={"due": "2026-01-20", "done": False},
    )
    overdue = find_overdue([(Path("x.md"), record)], as_of=date(2026, 2, 1))
    assert {o["check"] for o in overdue} == {"ship_check", "outcome_check"}


def test_real_example_records_have_exactly_the_two_seeded_overdue_decisions():
    records = load_records(BASE / "decisions")
    overdue = find_overdue(records, as_of=date.today())
    ids = {o["id"] for o in overdue}
    assert ids == {"DSG-0004", "DSG-0008"}
