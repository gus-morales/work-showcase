"""The Model Scope Document contract: the one design doc every pipeline
run is built from. It's a markdown file with a machine-readable YAML
header (parsed here) plus a free-text body (not parsed, humans read
it). `status` gates everything downstream: no stage runs against a
Model Scope that isn't `frozen`, the same way project 07's decision
records gate on their own lifecycle field instead of on memory.

A Model Scope names *what* the pipeline is building (objective,
target, metric, a feature-count cap) but deliberately carries no
warehouse-execution detail, that's the Data stage's job (see
`stage_io.py`). This file only validates that the design itself is
internally consistent, e.g. that `task_type='both'` actually has two
targets of the right kinds.
"""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

TaskType = Literal["classification", "regression", "both"]
TargetKind = Literal["classification", "regression"]
Status = Literal["draft", "signed_off", "frozen"]


class Project(BaseModel):
    name: str
    slug: str


class TargetSpec(BaseModel):
    name: str
    kind: TargetKind
    raw_field: str
    positive_label: str | None = None  # classification only, human-readable


class MetricSpec(BaseModel):
    target: str  # the TargetSpec.name this metric evaluates
    name: str
    goal: Literal["maximize", "minimize"]
    threshold: float


class ModelScope(BaseModel):
    project: Project
    objective: str
    problem: str
    task_type: TaskType
    targets: list[TargetSpec]
    metrics: list[MetricSpec]
    max_features: int
    data_sources: list[str]
    status: Status

    @model_validator(mode="after")
    def _check_consistency(self):
        if self.max_features < 1:
            raise ValueError("max_features must be >= 1")
        if not self.data_sources:
            raise ValueError("data_sources must list at least one source")

        kinds = [t.kind for t in self.targets]
        if self.task_type in ("classification", "regression"):
            if len(self.targets) != 1:
                raise ValueError(f"task_type='{self.task_type}' requires exactly one target, got {len(self.targets)}")
            if kinds[0] != self.task_type:
                raise ValueError(f"task_type='{self.task_type}' requires a target of the same kind, got kind='{kinds[0]}'")
        else:  # both
            if len(self.targets) != 2:
                raise ValueError(f"task_type='both' requires exactly two targets, got {len(self.targets)}")
            if sorted(kinds) != ["classification", "regression"]:
                raise ValueError(f"task_type='both' requires one classification and one regression target, got kinds={kinds}")

        names = [t.name for t in self.targets]
        if len(names) != len(set(names)):
            raise ValueError(f"target names must be unique, got {names}")

        if len(self.metrics) != len(self.targets):
            raise ValueError(f"expected one metric per target ({len(self.targets)}), got {len(self.metrics)}")
        metric_targets = sorted(m.target for m in self.metrics)
        if metric_targets != sorted(names):
            raise ValueError(f"metrics must reference each target exactly once: metrics cover {metric_targets}, targets are {sorted(names)}")

        return self

    def target(self, kind: TargetKind | None = None) -> TargetSpec:
        """The single target, or the target of the given kind for a
        task_type='both' scope. Raises if the request is ambiguous."""
        if kind is None:
            if len(self.targets) != 1:
                raise ValueError("scope has multiple targets; pass kind='classification'|'regression'")
            return self.targets[0]
        matches = [t for t in self.targets if t.kind == kind]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one target of kind='{kind}', found {len(matches)}")
        return matches[0]

    def metric_for(self, target_name: str) -> MetricSpec:
        matches = [m for m in self.metrics if m.target == target_name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one metric for target='{target_name}', found {len(matches)}")
        return matches[0]


def parse_model_scope_file(path: Path) -> ModelScope:
    """Parse a MODEL_SCOPE.md file's YAML frontmatter into a ModelScope.
    Same frontmatter convention as project 07's decision records: a
    '---' delimited block at the top of the file, prose after it."""
    text = path.read_text()
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing frontmatter delimiter ('---')")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: malformed frontmatter, expected '---' ... '---' at the top of the file")
    header = yaml.safe_load(parts[1]) or {}
    return ModelScope(**header)


def require_frozen(scope: ModelScope) -> None:
    """The one gate every stage checks before it runs: nothing touches
    a Model Scope that hasn't been through human sign-off and frozen."""
    if scope.status != "frozen":
        raise ValueError(
            f"Model Scope '{scope.project.slug}' has status='{scope.status}', "
            "the pipeline only runs against status='frozen'"
        )
