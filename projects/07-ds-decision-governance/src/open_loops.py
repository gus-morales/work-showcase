"""
CLI: scan decisions/ for records currently overdue on a monitoring
check, i.e. the check's due date has passed and it isn't marked done.
This is a live status check against whatever's in decisions/ right now,
not a report over historical data.

Run:
    python src/open_loops.py
"""
import sys
from datetime import date
from pathlib import Path

from schema import parse_decision_file, DecisionRecord

BASE = Path(__file__).resolve().parents[1]


def find_overdue(records: list[tuple[Path, DecisionRecord]], as_of: date) -> list[dict]:
    """Pure function: given parsed (path, record) pairs and a reference
    date, return one entry per overdue check (a record can appear twice,
    once for each check it's overdue on)."""
    overdue = []
    for path, record in records:
        if record.monitoring is None:
            continue
        ship = record.monitoring.ship_check
        if ship is not None and not ship.done and ship.due < as_of:
            overdue.append({
                "path": path, "id": record.id, "title": record.title,
                "check": "ship_check", "due": ship.due, "days_overdue": (as_of - ship.due).days,
            })
        outcome = record.monitoring.outcome_check
        if outcome is not None and not outcome.done and outcome.due < as_of:
            overdue.append({
                "path": path, "id": record.id, "title": record.title,
                "check": "outcome_check", "due": outcome.due, "days_overdue": (as_of - outcome.due).days,
            })
    return overdue


def load_records(decisions_dir: Path) -> list[tuple[Path, DecisionRecord]]:
    records = []
    for path in sorted(decisions_dir.rglob("*.md")):
        frontmatter, _ = parse_decision_file(path)
        records.append((path, DecisionRecord(**frontmatter)))
    return records


def main():
    decisions_dir = BASE / "decisions"
    records = load_records(decisions_dir)
    overdue = find_overdue(records, date.today())

    if not overdue:
        print("No overdue checks.")
        return 0

    overdue.sort(key=lambda o: -o["days_overdue"])
    print(f"{len(overdue)} overdue check(s):\n")
    for o in overdue:
        print(f"{o['id']} ({o['path'].relative_to(BASE)}): {o['check']} was due {o['due']}, "
              f"{o['days_overdue']} day(s) ago - \"{o['title']}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
