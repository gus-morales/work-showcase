"""The conductor. Chains EDA -> Data -> Features -> Training ->
Validation for one run directory, and between every stage, checks the
stage's output against the next stage's declared contract
(`stage_io.verify_*`) before advancing. On any mismatch it stops
immediately, naming the stage and the exact field that didn't hold. It
never skips a stage and never fabricates an artifact to keep the chain
moving, the two things the real framework this project is modeled on
is explicit about never doing either.
"""
import argparse
from pathlib import Path

from bindings import load_bindings, load_feature_spec
from model_scope import parse_model_scope_file, require_frozen
from stage_io import (
    ContractViolation,
    verify_data_stage_output,
    verify_eda_output,
    verify_feature_stage_output,
    verify_training_stage_output,
    verify_validation_stage_output,
)
from stages import data_prep, eda, feature_engineering, training, validation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(run_dir: Path, verbose: bool = True) -> dict:
    run_dir = Path(run_dir).resolve()
    scope = parse_model_scope_file(run_dir / "MODEL_SCOPE.md")
    require_frozen(scope)  # the one gate: nothing below runs against a Scope that isn't frozen
    bindings = load_bindings(run_dir / "bindings.yaml")
    feature_spec = load_feature_spec(run_dir / "feature_spec.yaml")

    def log(msg):
        if verbose:
            print(msg)

    log(f"Model Scope '{scope.project.slug}' frozen, task_type={scope.task_type}. Running pipeline.")

    log("[1/5] EDA")
    eda_output = eda.run(scope, bindings, run_dir)
    verify_eda_output(eda_output)

    log("[2/5] Data")
    data_output = data_prep.run(scope, bindings, run_dir)
    verify_data_stage_output(data_output, scope)
    log(f"      training_data: {data_output.row_count:,} rows")

    log("[3/5] Feature Engineering")
    feature_output = feature_engineering.run(scope, data_output, feature_spec, run_dir)
    verify_feature_stage_output(feature_output, data_output, scope)
    log(f"      features: {len(feature_output.feature_cols)} (cap {scope.max_features})")

    log("[4/5] Training")
    training_output = training.run(scope, feature_output, data_output.manifest_path, PROJECT_ROOT, run_dir)
    verify_training_stage_output(training_output, feature_output, scope)
    for name, value in training_output.metric_values.items():
        log(f"      {name}: {scope.metric_for(name).name}={value:.4f}")

    log("[5/5] Validation")
    validation_output = validation.run(scope, feature_output, training_output, data_output.manifest_path, PROJECT_ROOT, run_dir)
    verify_validation_stage_output(validation_output, scope)
    log(f"      overall: {'PASS' if validation_output.passed else 'FAIL'}")

    return {
        "scope": scope, "eda": eda_output, "data": data_output,
        "features": feature_output, "training": training_output, "validation": validation_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the ML pipeline for one run directory.")
    parser.add_argument("--run-dir", required=True, help="e.g. runs/churn/run-001")
    args = parser.parse_args()

    try:
        run_pipeline(Path(args.run_dir))
    except ContractViolation as exc:
        print(f"\nSTOPPED at stage '{exc.stage}': {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
