"""
The record contract: what a decision file has to contain to be valid.
A decision is a markdown file with YAML frontmatter (the structured
fields below) plus a free-text body. `validate_file()` parses one file
and returns every reason it fails, if any; an empty list means it's
valid. This is the same role a CI schema check plays on a real
pull-request-based governance repo, run here as a standalone script
instead of a merge gate.
"""
import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

ArtifactType = Literal[
    "dashboard_change", "pipeline_change", "experiment_rollout",
    "model_launch", "metric_definition_change", "deprecation",
]
DomainTag = Literal[
    "product_analytics", "search_ranking", "marketing",
    "customer_support", "operations", "infrastructure",
]
ImpactLevel = Literal["low", "medium", "high"]
Status = Literal["draft", "approved", "shipped", "closed", "reverted", "abandoned"]
Outcome = Literal["keep", "iterate", "rollback"]


class Dates(BaseModel):
    proposed: date
    approved: date | None = None
    shipped: date | None = None
    resolved: date | None = None


class ShipCheck(BaseModel):
    due: date
    done: bool


class OutcomeCheck(BaseModel):
    due: date
    done: bool
    outcome: Outcome | None = None


class Monitoring(BaseModel):
    ship_check: ShipCheck | None = None
    outcome_check: OutcomeCheck | None = None


class DecisionRecord(BaseModel):
    id: str
    title: str
    artifact_type: ArtifactType
    domain: DomainTag
    impact_level: ImpactLevel
    status: Status
    author: str
    dates: Dates
    monitoring: Monitoring | None = None
    reviewers: list[str] = []

    @model_validator(mode="after")
    def _check_consistency(self):
        if self.status == "draft":
            if self.dates.approved or self.dates.shipped or self.dates.resolved:
                raise ValueError("status='draft' cannot have approved/shipped/resolved dates")
            return self

        if self.status == "abandoned":
            if self.dates.approved or self.dates.shipped:
                raise ValueError("status='abandoned' cannot have approved/shipped dates")
            return self

        # approved, shipped, closed, reverted all went through approval.
        if self.dates.approved is None:
            raise ValueError(f"status={self.status} requires an approved date")

        shipped_or_later = self.status in ("shipped", "closed", "reverted")
        if shipped_or_later and self.dates.shipped is None:
            raise ValueError(f"status={self.status} requires a shipped date")

        expects_ship_check = shipped_or_later and self.impact_level in ("medium", "high")
        if expects_ship_check and (self.monitoring is None or self.monitoring.ship_check is None):
            raise ValueError(f"impact_level={self.impact_level} requires a ship_check once shipped")

        if shipped_or_later and (self.monitoring is None or self.monitoring.outcome_check is None):
            raise ValueError("a shipped decision must have an outcome_check")

        if self.status in ("closed", "reverted"):
            if self.dates.resolved is None:
                raise ValueError(f"status={self.status} requires a resolved date")
            outcome = self.monitoring.outcome_check.outcome if self.monitoring and self.monitoring.outcome_check else None
            if outcome is None:
                raise ValueError(f"status={self.status} must have an outcome_check.outcome")
            if self.status == "reverted" and outcome != "rollback":
                raise ValueError("status='reverted' must have outcome='rollback'")
            if self.status == "closed" and outcome == "rollback":
                raise ValueError("outcome='rollback' must have status='reverted', not 'closed'")

        return self


def parse_decision_file(path: Path) -> tuple[dict, str]:
    """Split a decision file into its YAML frontmatter and markdown body."""
    text = path.read_text()
    if not text.startswith("---"):
        raise ValueError("missing frontmatter delimiter ('---')")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("malformed frontmatter: expected '---' ... '---' at the top of the file")
    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    return frontmatter, body


def has_rollback_plan(body: str) -> bool:
    """A '## Rollback plan' section exists and has real content under it,
    not just the template's guidance comment."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "## rollback plan":
            rest = "\n".join(lines[i + 1:])
            rest = rest.split("\n## ")[0]
            rest = re.sub(r"<!--.*?-->", "", rest, flags=re.DOTALL).strip()
            return len(rest) > 0
    return False


def validate_file(path: Path, routing: dict) -> list[str]:
    """Validate one decision file. Returns every failure reason; an
    empty list means the record is valid."""
    try:
        frontmatter, body = parse_decision_file(path)
        record = DecisionRecord(**frontmatter)
    except Exception as exc:
        return [str(exc)]

    errors = []
    # Reviewers get assigned as part of approval, so a draft (still being
    # written) or an abandoned record (never reached approval) isn't held
    # to the reviewer-count minimum yet.
    if record.status not in ("draft", "abandoned"):
        required_reviewers = routing[record.impact_level]["approvers"]
        if len(record.reviewers) < required_reviewers:
            errors.append(
                f"impact_level={record.impact_level} requires >= {required_reviewers} reviewers, "
                f"got {len(record.reviewers)}"
            )
    if record.impact_level == "high" and not has_rollback_plan(body):
        errors.append("impact_level=high requires a non-empty '## Rollback plan' section")

    return errors


def validate_directory(decisions_dir: Path, routing: dict) -> dict[Path, list[str]]:
    """Validate every *.md file under decisions_dir. Returns a map of
    path -> errors, only for files that failed."""
    failures = {}
    for path in sorted(decisions_dir.rglob("*.md")):
        errors = validate_file(path, routing)
        if errors:
            failures[path] = errors
    return failures


def load_routing(routing_path: Path) -> dict:
    return yaml.safe_load(routing_path.read_text())
