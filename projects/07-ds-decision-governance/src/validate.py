"""
CLI: validate every decision record under decisions/. Exits non-zero if
any record fails, the same check a CI job would run against a pull
request in a real repo, run here as a standalone script instead.

Run:
    python src/validate.py
"""
import sys
from pathlib import Path

from schema import load_routing, validate_directory

BASE = Path(__file__).resolve().parents[1]


def main():
    routing = load_routing(BASE / "routing.yaml")
    decisions_dir = BASE / "decisions"
    failures = validate_directory(decisions_dir, routing)

    n_total = len(list(decisions_dir.rglob("*.md")))
    n_failed = len(failures)

    if not failures:
        print(f"{n_total} / {n_total} records valid.")
        return 0

    print(f"{n_total - n_failed} / {n_total} records valid. Failures:")
    for path, errors in failures.items():
        print(f"\n{path.relative_to(BASE)}")
        for error in errors:
            print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
