"""The fixed seams between pipeline stages: what each stage promises to
produce, and a `verify_*` function per stage that actually opens the
artifact and checks the promise holds, not just that a Pydantic model
with the right field names was returned. A stage can satisfy its own
output schema and still hand the next stage something broken, e.g. a
`training_data.parquet` that doesn't actually contain the column it
claims is the identifier. The `verify_*` functions are what catch
that, and they're what the orchestrator calls between every stage.

Every failure raises `ContractViolation` with a message naming the
stage and the specific field that didn't hold, the same
owner-addressed-error idea project 07's `open_loops.py` and
`validate.py` use for a broken decision record.
"""
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from model_scope import ModelScope


class ContractViolation(Exception):
    """A stage's declared output doesn't match what the next stage's
    contract requires. Carries the stage name so the orchestrator can
    stop with an owner-addressed message instead of a bare traceback."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


# --- ① EDA ------------------------------------------------------------

class EDAOutput(BaseModel):
    sample_profile_path: Path
    target_exploration_path: Path


def verify_eda_output(output: EDAOutput) -> None:
    for field, path in (("sample_profile_path", output.sample_profile_path),
                         ("target_exploration_path", output.target_exploration_path)):
        if not path.exists():
            raise ContractViolation("eda", f"declared {field}={path} but the file doesn't exist")


# --- ② Data -------------------------------------------------------------

class DataStageOutput(BaseModel):
    training_data_path: Path
    data_dictionary_path: Path
    manifest_path: Path
    identifier_col: str
    row_count: int


def verify_data_stage_output(output: DataStageOutput, scope: ModelScope) -> None:
    if not output.training_data_path.exists():
        raise ContractViolation("data", f"declared training_data_path={output.training_data_path} but the file doesn't exist")

    df = pd.read_parquet(output.training_data_path)

    if output.identifier_col not in df.columns:
        raise ContractViolation(
            "data",
            f"declared identifier_col='{output.identifier_col}' but training_data.parquet's "
            f"columns are {list(df.columns)}",
        )
    if df[output.identifier_col].isna().any():
        raise ContractViolation("data", f"identifier_col='{output.identifier_col}' has null values")
    if df[output.identifier_col].duplicated().any():
        raise ContractViolation("data", f"identifier_col='{output.identifier_col}' has duplicate values, one row per example is required")

    for target in scope.targets:
        if target.name not in df.columns:
            raise ContractViolation(
                "data",
                f"Model Scope declares target '{target.name}' but training_data.parquet's "
                f"columns are {list(df.columns)}",
            )
        if df[target.name].isna().any():
            raise ContractViolation("data", f"target '{target.name}' has null values in training_data.parquet")

    if len(df) != output.row_count:
        raise ContractViolation(
            "data",
            f"declared row_count={output.row_count} but training_data.parquet has {len(df)} rows",
        )


# --- ③ Feature engineering -----------------------------------------------

class FeatureStageOutput(BaseModel):
    feature_matrix_path: Path
    feature_taxonomy_path: Path
    validation_report_path: Path
    identifier_col: str
    feature_cols: list[str]


def verify_feature_stage_output(output: FeatureStageOutput, data_output: DataStageOutput, scope: ModelScope) -> None:
    if not output.feature_matrix_path.exists():
        raise ContractViolation("features", f"declared feature_matrix_path={output.feature_matrix_path} but the file doesn't exist")

    df = pd.read_parquet(output.feature_matrix_path)

    if output.identifier_col != data_output.identifier_col:
        raise ContractViolation(
            "features",
            f"identifier_col='{output.identifier_col}' doesn't match the Data stage's "
            f"identifier_col='{data_output.identifier_col}'",
        )
    if output.identifier_col not in df.columns:
        raise ContractViolation("features", f"declared identifier_col='{output.identifier_col}' but it's not a column in features.parquet")

    missing_features = [c for c in output.feature_cols if c not in df.columns]
    if missing_features:
        raise ContractViolation("features", f"declared feature_cols include {missing_features}, not present in features.parquet")

    target_names = {t.name for t in scope.targets}
    leaked = target_names & set(output.feature_cols)
    if leaked:
        raise ContractViolation("features", f"target column(s) {sorted(leaked)} appear in feature_cols, that's target leakage")

    if len(output.feature_cols) > scope.max_features:
        raise ContractViolation(
            "features",
            f"{len(output.feature_cols)} features exceeds the Model Scope's max_features={scope.max_features}",
        )

    for target in scope.targets:
        if target.name not in df.columns:
            raise ContractViolation("features", f"features.parquet is missing target column '{target.name}', Training needs it to fit against")


# --- ④ Training ------------------------------------------------------------

class TrainingStageOutput(BaseModel):
    model_paths: dict[str, Path]              # target name -> model file
    feature_importance_path: Path
    selected_features_path: Path
    train_report_path: Path
    selected_features: dict[str, list[str]]   # target name -> its selected features
    metric_values: dict[str, float]           # target name -> metric value on the holdout split


def verify_training_stage_output(output: TrainingStageOutput, feature_output: FeatureStageOutput, scope: ModelScope) -> None:
    target_names = {t.name for t in scope.targets}
    if set(output.model_paths) != target_names:
        raise ContractViolation("training", f"model_paths keys {sorted(output.model_paths)} don't match the Model Scope's targets {sorted(target_names)}")

    for name, path in output.model_paths.items():
        if not path.exists():
            raise ContractViolation("training", f"declared model_paths['{name}']={path} but the file doesn't exist")

    for name, features in output.selected_features.items():
        unknown = set(features) - set(feature_output.feature_cols)
        if unknown:
            raise ContractViolation("training", f"selected_features['{name}'] include {sorted(unknown)}, not present in the Feature stage's feature_cols")
        if len(features) > scope.max_features:
            raise ContractViolation("training", f"selected_features['{name}'] has {len(features)} features, exceeds max_features={scope.max_features}")
        if not features:
            raise ContractViolation("training", f"selected_features['{name}'] is empty, the reduction step dropped every feature")

    import math
    for name, value in output.metric_values.items():
        if math.isnan(value) or math.isinf(value):
            raise ContractViolation("training", f"metric_values['{name}']={value} is not a finite number")
    if set(output.metric_values) != target_names:
        raise ContractViolation("training", f"metric_values keys {sorted(output.metric_values)} don't match the Model Scope's targets {sorted(target_names)}")


# --- ⑤ Validation ----------------------------------------------------------

class TargetValidationResult(BaseModel):
    metric_value: float
    threshold: float
    passed: bool


class ValidationStageOutput(BaseModel):
    criteria_path: Path
    report_path: Path
    results: dict[str, TargetValidationResult]   # target name -> result
    passed: bool                                   # overall: all targets pass


def verify_validation_stage_output(output: ValidationStageOutput, scope: ModelScope) -> None:
    target_names = {t.name for t in scope.targets}
    if set(output.results) != target_names:
        raise ContractViolation("validation", f"results keys {sorted(output.results)} don't match the Model Scope's targets {sorted(target_names)}")

    for name, result in output.results.items():
        metric = scope.metric_for(name)
        expected_pass = result.metric_value >= result.threshold if metric.goal == "maximize" else result.metric_value <= result.threshold
        if result.passed != expected_pass:
            raise ContractViolation(
                "validation",
                f"results['{name}'].passed={result.passed} is inconsistent with metric_value={result.metric_value}, "
                f"threshold={result.threshold}, goal='{metric.goal}'",
            )
        if abs(result.threshold - metric.threshold) > 1e-9:
            raise ContractViolation(
                "validation",
                f"results['{name}'].threshold={result.threshold} doesn't match the Model Scope's threshold={metric.threshold}",
            )

    expected_overall = all(r.passed for r in output.results.values())
    if output.passed != expected_overall:
        raise ContractViolation("validation", f"passed={output.passed} is inconsistent with per-target results {output.results}")
