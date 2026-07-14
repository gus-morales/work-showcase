"""Unit tests for the record contract: the real example records under
decisions/ all pass, and each rule the schema is supposed to enforce
actually catches a record that breaks it."""
from pathlib import Path

from schema import validate_file, validate_directory, load_routing

BASE = Path(__file__).resolve().parents[1]
ROUTING = load_routing(BASE / "routing.yaml")

VALID_LOW = """---
id: DSG-9001
title: "Example"
artifact_type: dashboard_change
domain: product_analytics
impact_level: low
status: closed
author: "Tester"
dates:
  proposed: 2026-01-01
  approved: 2026-01-02
  shipped: 2026-01-05
  resolved: 2026-02-01
monitoring:
  outcome_check:
    due: 2026-02-01
    done: true
    outcome: keep
reviewers: ["A"]
---

## What changed
Something.
"""

VALID_HIGH = """---
id: DSG-9003
title: "Example"
artifact_type: model_launch
domain: product_analytics
impact_level: high
status: closed
author: "Tester"
dates:
  proposed: 2026-01-01
  approved: 2026-01-02
  shipped: 2026-01-05
  resolved: 2026-02-01
monitoring:
  ship_check:
    due: 2026-01-12
    done: true
  outcome_check:
    due: 2026-02-01
    done: true
    outcome: keep
reviewers: ["A", "B", "C"]
---

## Rollback plan
Revert the config change.
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "record.md"
    path.write_text(text)
    return path


def test_all_example_records_are_valid():
    failures = validate_directory(BASE / "decisions", ROUTING)
    assert failures == {}


def test_valid_low_impact_record_passes(tmp_path):
    assert validate_file(_write(tmp_path, VALID_LOW), ROUTING) == []


def test_valid_high_impact_record_passes(tmp_path):
    assert validate_file(_write(tmp_path, VALID_HIGH), ROUTING) == []


def test_shipped_medium_without_ship_check_fails(tmp_path):
    text = VALID_LOW.replace("impact_level: low", "impact_level: medium")
    errors = validate_file(_write(tmp_path, text), ROUTING)
    assert len(errors) >= 1


def test_high_impact_without_rollback_plan_section_fails(tmp_path):
    text = VALID_HIGH.split("---\n\n## Rollback plan")[0] + "---\n"
    errors = validate_file(_write(tmp_path, text), ROUTING)
    assert any("Rollback plan" in e for e in errors)


def test_insufficient_reviewers_fails(tmp_path):
    text = VALID_HIGH.replace('reviewers: ["A", "B", "C"]', 'reviewers: ["A"]')
    errors = validate_file(_write(tmp_path, text), ROUTING)
    assert any("reviewers" in e for e in errors)


def test_reverted_status_requires_rollback_outcome(tmp_path):
    text = VALID_LOW.replace("status: closed", "status: reverted").replace(
        "outcome: keep", "outcome: iterate"
    )
    errors = validate_file(_write(tmp_path, text), ROUTING)
    assert len(errors) >= 1


def test_closed_status_cannot_have_rollback_outcome(tmp_path):
    text = VALID_LOW.replace("outcome: keep", "outcome: rollback")
    errors = validate_file(_write(tmp_path, text), ROUTING)
    assert len(errors) >= 1


def test_closed_status_requires_resolved_date(tmp_path):
    text = VALID_LOW.replace("  resolved: 2026-02-01\n", "  resolved: null\n")
    errors = validate_file(_write(tmp_path, text), ROUTING)
    assert len(errors) >= 1


def test_draft_status_skips_reviewer_check(tmp_path):
    text = """---
id: DSG-9002
title: "Example draft"
artifact_type: model_launch
domain: product_analytics
impact_level: high
status: draft
author: "Tester"
dates:
  proposed: 2026-01-01
monitoring: null
reviewers: []
---

## Rollback plan
Not shipped yet, but the plan: revert the config change.
"""
    assert validate_file(_write(tmp_path, text), ROUTING) == []
