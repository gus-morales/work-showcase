"""Scaffolds a new run directory: `runs/<slug>/<run-id>/MODEL_SCOPE.md`
from a template, status `draft`. Mirrors project 07's `new_decision.py`
in spirit: it fills in what it can from the command line, everything
else (the target definitions, the metric, sign-off) gets filled in by
hand before the pipeline will run against it, `require_frozen()`
blocks that until `status: frozen` is set.
"""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = """\
---
project:
  name: "{name}"
  slug: {slug}
objective: "{objective}"
problem: "{problem}"
task_type: {task_type}
targets:
  - name: TBD
    kind: {first_kind}
    raw_field: TBD
metrics:
  - target: TBD
    name: TBD
    goal: maximize
    threshold: 0.0
max_features: {max_features}
data_sources: [TBD]
status: draft
---

## Introduction

TBD: what this model is for, in a sentence or two.

## Model Scope

TBD: what this run covers and what it defers.

## Target Proposal

TBD: fill in after the EDA stage runs, pointing at
`artifacts/eda/target_exploration.json`.

## Data Sources for Feature Engineering

TBD: pointer to this run's `bindings.yaml` and `feature_spec.yaml`.

## Model Evaluation

TBD: why this metric and this threshold.

## Sign-off

TBD: who reviewed this, and the date `status` moved to `frozen`.
"""


def main():
    parser = argparse.ArgumentParser(description="Scaffold a new pipeline run.")
    parser.add_argument("--slug", required=True, help="problem slug, e.g. churn")
    parser.add_argument("--run-id", required=True, help="e.g. run-001")
    parser.add_argument("--name", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--task-type", choices=["classification", "regression", "both"], default="classification")
    parser.add_argument("--max-features", type=int, default=10)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "runs" / args.slug / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()

    first_kind = "regression" if args.task_type == "regression" else "classification"
    content = TEMPLATE.format(
        name=args.name, slug=args.slug, objective=args.objective, problem=args.problem,
        task_type=args.task_type, first_kind=first_kind, max_features=args.max_features,
    )
    (run_dir / "MODEL_SCOPE.md").write_text(content)
    print(f"Wrote {run_dir / 'MODEL_SCOPE.md'} (status: draft)")
    print("Fill in targets/metrics/data_sources, add bindings.yaml + feature_spec.yaml, "
          "then set status: frozen once signed off before running the orchestrator.")


if __name__ == "__main__":
    main()
