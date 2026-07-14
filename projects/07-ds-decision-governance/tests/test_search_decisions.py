"""Unit tests for the ranking logic in search_decisions.py: a
near-duplicate decision should outrank an unrelated one, and the real
example decisions should surface the two known deprecation records
together for a deprecation-shaped query."""
from pathlib import Path

from schema import DecisionRecord
from search_decisions import load_decisions, search

BASE = Path(__file__).resolve().parents[1]


def _record(id: str, title: str, artifact_type: str, domain: str = "product_analytics") -> DecisionRecord:
    return DecisionRecord(
        id=id,
        title=title,
        artifact_type=artifact_type,
        domain=domain,
        impact_level="low",
        status="draft",
        author="Tester",
        dates={"proposed": "2026-01-01"},
    )


def _why_body(text: str) -> str:
    return f"\n## Why\n\n{text}\n"


def test_near_duplicate_decision_outranks_unrelated():
    near_duplicate = _record("DSG-1", "Decommission the legacy scoring model", "deprecation")
    unrelated = _record("DSG-2", "Switch the campaign dashboard to weekly cohorts", "dashboard_change")
    records = [
        (Path("a.md"), near_duplicate, _why_body("Old model is superseded and nobody reads its scores anymore.")),
        (Path("b.md"), unrelated, _why_body("Weekly cohorts are easier for the marketing team to act on.")),
    ]

    ranked = search("should we retire an old model nobody uses anymore", records)

    assert ranked[0][1].id == "DSG-1"
    assert ranked[0][2] > ranked[1][2]


def test_top_n_limits_results():
    records = [
        (Path(f"{i}.md"), _record(f"DSG-{i}", f"Decision {i}", "dashboard_change"), _why_body(f"reason {i}"))
        for i in range(5)
    ]
    ranked = search("reason 2", records, top_n=2)
    assert len(ranked) == 2


def test_empty_records_returns_empty():
    assert search("anything", []) == []


def test_real_example_decisions_surface_both_deprecations():
    records = load_decisions(BASE / "decisions")
    ranked = search("should we retire an old model or endpoint nobody uses anymore", records, top_n=3)
    top_ids = {record.id for _, record, _ in ranked}
    assert {"DSG-0008", "DSG-0003"}.issubset(top_ids)
