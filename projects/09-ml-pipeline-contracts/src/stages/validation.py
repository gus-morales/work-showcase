"""Stage ⑤ Validation. Reloads each target's selected-feature holdout
split, scores the trained model against it, and checks the result
against the Model Scope's declared metric and threshold. Doesn't
retrain anything and doesn't touch the feature-reduction history,
its only job is an independent read of "does this clear the bar the
Model Scope set," per target, plus one overall pass/fail the
orchestrator surfaces to a human before Deployment would run.
"""
import json
import pickle
from pathlib import Path

import pandas as pd

from model_scope import ModelScope
from stage_io import FeatureStageOutput, TrainingStageOutput, TargetValidationResult, ValidationStageOutput
from stages.training import _score, _time_split


def run(scope: ModelScope, feature_output: FeatureStageOutput, training_output: TrainingStageOutput,
        manifest_path: Path, project_root: Path, run_dir: Path) -> ValidationStageOutput:
    manifest = json.loads(manifest_path.read_text())
    feature_matrix = pd.read_parquet(feature_output.feature_matrix_path)
    _, test_df = _time_split(feature_matrix, manifest, project_root)

    results = {}
    criteria_lines = ["# Validation criteria", ""]
    report_lines = ["# Validation report", ""]

    for target in scope.targets:
        metric = scope.metric_for(target.name)
        with open(training_output.model_paths[target.name], "rb") as f:
            model = pickle.load(f)
        features = training_output.selected_features[target.name]
        metric_value = _score(target.kind, metric.name, model, test_df[features], test_df[target.name])
        passed = metric_value >= metric.threshold if metric.goal == "maximize" else metric_value <= metric.threshold
        results[target.name] = TargetValidationResult(metric_value=round(metric_value, 5), threshold=metric.threshold, passed=passed)

        criteria_lines.append(f"- `{target.name}`: {metric.name} must {metric.goal} to {'>=' if metric.goal == 'maximize' else '<='} {metric.threshold}")
        report_lines.append(f"## {target.name}\n\n- {metric.name} on holdout: {metric_value:.4f}\n- threshold: {metric.threshold} (goal: {metric.goal})\n- **{'PASS' if passed else 'FAIL'}**\n")

    overall_passed = all(r.passed for r in results.values())
    report_lines.append(f"## Overall\n\n**{'PASS' if overall_passed else 'FAIL'}** — {sum(r.passed for r in results.values())}/{len(results)} targets clear their threshold.")

    artifacts_dir = run_dir / "artifacts" / "validation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    criteria_path = artifacts_dir / "criteria.md"
    report_path = artifacts_dir / "report.md"
    criteria_path.write_text("\n".join(criteria_lines) + "\n")
    report_path.write_text("\n".join(report_lines) + "\n")

    return ValidationStageOutput(criteria_path=criteria_path, report_path=report_path, results=results, passed=overall_passed)
