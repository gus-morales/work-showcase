"""Runs EDA and Data for real against the churn run, then simulates the
single most common way a Feature Engineering stage breaks its contract
with Training: the target column rides along as a "feature" because
whoever built the join didn't drop it. Nothing here is paraphrased,
`main()` prints the actual `ContractViolation` the orchestrator would
stop on, and `runs/broken-example/run-001/` is left with a copy of
that real output.

Run: `python demo_broken_contract.py` from `src/`.
"""
from pathlib import Path

import pandas as pd

from bindings import load_bindings
from model_scope import parse_model_scope_file
from stage_io import ContractViolation, FeatureStageOutput, verify_data_stage_output, verify_feature_stage_output
from stages import data_prep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHURN_RUN = PROJECT_ROOT / "runs" / "churn" / "run-001"
BROKEN_RUN = PROJECT_ROOT / "runs" / "broken-example" / "run-001"


def main():
    scope = parse_model_scope_file(CHURN_RUN / "MODEL_SCOPE.md")
    bindings = load_bindings(CHURN_RUN / "bindings.yaml")

    # Stages ① and ② run for real, against the real churn Model Scope,
    # so the run directory this failure lives in has a genuine, valid
    # Data-stage artifact behind it.
    BROKEN_RUN.mkdir(parents=True, exist_ok=True)
    data_output = data_prep.run(scope, bindings, BROKEN_RUN)
    verify_data_stage_output(data_output, scope)  # this part is clean

    # Stage ③, simulated broken: someone built the feature matrix by
    # joining training_data straight in, target column included, and
    # declared it as a feature without noticing.
    training_data = pd.read_parquet(data_output.training_data_path)
    broken_features = training_data.copy()
    broken_features["avg_logins_pre_decision"] = 1.5  # a real-looking feature, present
    artifacts_dir = BROKEN_RUN / "artifacts" / "features"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    broken_path = artifacts_dir / "features.parquet"
    broken_features.to_parquet(broken_path, index=False)

    broken_output = FeatureStageOutput(
        feature_matrix_path=broken_path,
        feature_taxonomy_path=artifacts_dir / "feature_taxonomy.md",  # not written, doesn't matter for this check
        validation_report_path=artifacts_dir / "validation.md",
        identifier_col=data_output.identifier_col,
        feature_cols=["avg_logins_pre_decision", "churned_next_30d"],  # the target, declared as a feature
    )

    output_lines = [
        "# Caught contract violation (real output, not paraphrased)",
        "",
        "Stages 1-2 (EDA, Data) ran for real against `runs/churn/run-001`'s Model Scope and",
        "produced a valid `training_data.parquet`. Stage 3 (Feature Engineering) is simulated",
        "here as broken: its declared `feature_cols` include `churned_next_30d`, the target",
        "itself, the single most common way a feature matrix leaks its own label.",
        "",
        "```",
    ]
    print("Verifying a Feature stage output that declares the target as a feature...")
    try:
        verify_feature_stage_output(broken_output, data_output, scope)
        print("no violation raised (unexpected)")
        output_lines.append("no violation raised (unexpected)")
    except ContractViolation as exc:
        message = f"ContractViolation: {exc}"
        print(message)
        output_lines.append(message)
    output_lines.append("```")
    output_lines.append("")
    output_lines.append("The orchestrator stops here: Training never runs against this feature matrix.")

    (BROKEN_RUN / "CAUGHT_VIOLATION.md").write_text("\n".join(output_lines) + "\n")
    print(f"\nWrote {BROKEN_RUN / 'CAUGHT_VIOLATION.md'}")


if __name__ == "__main__":
    main()
