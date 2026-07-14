import pandas as pd
import pytest

from model_scope import ModelScope
from stage_io import (
    ContractViolation,
    DataStageOutput,
    FeatureStageOutput,
    TargetValidationResult,
    TrainingStageOutput,
    ValidationStageOutput,
    verify_data_stage_output,
    verify_feature_stage_output,
    verify_training_stage_output,
    verify_validation_stage_output,
)


@pytest.fixture
def scope():
    return ModelScope(
        project={"name": "Test", "slug": "test"},
        objective="obj", problem="prob", task_type="classification",
        targets=[{"name": "churned", "kind": "classification", "raw_field": "churned"}],
        metrics=[{"target": "churned", "name": "pr_auc", "goal": "maximize", "threshold": 0.3}],
        max_features=5, data_sources=["usage_events"], status="frozen",
    )


# --- Data stage --------------------------------------------------------

def test_verify_data_stage_output_passes_for_clean_artifact(tmp_path, scope):
    df = pd.DataFrame({"customer_id": ["a", "b", "c"], "churned": [0, 1, 0]})
    path = tmp_path / "training_data.parquet"
    df.to_parquet(path)
    output = DataStageOutput(training_data_path=path, data_dictionary_path=path, manifest_path=path,
                              identifier_col="customer_id", row_count=3)
    verify_data_stage_output(output, scope)  # no raise


def test_verify_data_stage_output_catches_identifier_col_mismatch(tmp_path, scope):
    df = pd.DataFrame({"cust_id": ["a", "b", "c"], "churned": [0, 1, 0]})  # declared identifier isn't a real column
    path = tmp_path / "training_data.parquet"
    df.to_parquet(path)
    output = DataStageOutput(training_data_path=path, data_dictionary_path=path, manifest_path=path,
                              identifier_col="customer_id", row_count=3)
    with pytest.raises(ContractViolation, match="identifier_col='customer_id'"):
        verify_data_stage_output(output, scope)


def test_verify_data_stage_output_catches_missing_target(tmp_path, scope):
    df = pd.DataFrame({"customer_id": ["a", "b", "c"]})  # no 'churned' column at all
    path = tmp_path / "training_data.parquet"
    df.to_parquet(path)
    output = DataStageOutput(training_data_path=path, data_dictionary_path=path, manifest_path=path,
                              identifier_col="customer_id", row_count=3)
    with pytest.raises(ContractViolation, match="target 'churned'"):
        verify_data_stage_output(output, scope)


def test_verify_data_stage_output_catches_duplicate_identifiers(tmp_path, scope):
    df = pd.DataFrame({"customer_id": ["a", "a", "c"], "churned": [0, 1, 0]})
    path = tmp_path / "training_data.parquet"
    df.to_parquet(path)
    output = DataStageOutput(training_data_path=path, data_dictionary_path=path, manifest_path=path,
                              identifier_col="customer_id", row_count=3)
    with pytest.raises(ContractViolation, match="duplicate values"):
        verify_data_stage_output(output, scope)


def test_verify_data_stage_output_catches_wrong_row_count(tmp_path, scope):
    df = pd.DataFrame({"customer_id": ["a", "b", "c"], "churned": [0, 1, 0]})
    path = tmp_path / "training_data.parquet"
    df.to_parquet(path)
    output = DataStageOutput(training_data_path=path, data_dictionary_path=path, manifest_path=path,
                              identifier_col="customer_id", row_count=99)
    with pytest.raises(ContractViolation, match="row_count=99"):
        verify_data_stage_output(output, scope)


# --- Feature stage -------------------------------------------------------

def _data_output(tmp_path):
    df = pd.DataFrame({"customer_id": ["a", "b", "c"], "churned": [0, 1, 0]})
    path = tmp_path / "training_data.parquet"
    df.to_parquet(path)
    return DataStageOutput(training_data_path=path, data_dictionary_path=path, manifest_path=path,
                            identifier_col="customer_id", row_count=3)


def test_verify_feature_stage_output_catches_target_leakage(tmp_path, scope):
    data_output = _data_output(tmp_path)
    df = pd.DataFrame({"customer_id": ["a", "b", "c"], "churned": [0, 1, 0], "avg_logins": [1.0, 2.0, 3.0]})
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    output = FeatureStageOutput(feature_matrix_path=path, feature_taxonomy_path=path, validation_report_path=path,
                                 identifier_col="customer_id", feature_cols=["avg_logins", "churned"])
    with pytest.raises(ContractViolation, match="target leakage"):
        verify_feature_stage_output(output, data_output, scope)


def test_verify_feature_stage_output_catches_max_features_exceeded(tmp_path, scope):
    data_output = _data_output(tmp_path)
    cols = [f"f{i}" for i in range(7)]  # scope's max_features is 5
    df = pd.DataFrame({"customer_id": ["a", "b", "c"], "churned": [0, 1, 0], **{c: [1, 2, 3] for c in cols}})
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    output = FeatureStageOutput(feature_matrix_path=path, feature_taxonomy_path=path, validation_report_path=path,
                                 identifier_col="customer_id", feature_cols=cols)
    with pytest.raises(ContractViolation, match="exceeds the Model Scope's max_features"):
        verify_feature_stage_output(output, data_output, scope)


def test_verify_feature_stage_output_catches_missing_declared_column(tmp_path, scope):
    data_output = _data_output(tmp_path)
    df = pd.DataFrame({"customer_id": ["a", "b", "c"], "churned": [0, 1, 0]})
    path = tmp_path / "features.parquet"
    df.to_parquet(path)
    output = FeatureStageOutput(feature_matrix_path=path, feature_taxonomy_path=path, validation_report_path=path,
                                 identifier_col="customer_id", feature_cols=["avg_logins"])  # not actually in the file
    with pytest.raises(ContractViolation, match="not present in features.parquet"):
        verify_feature_stage_output(output, data_output, scope)


# --- Training stage --------------------------------------------------------

def test_verify_training_stage_output_catches_unknown_selected_feature(tmp_path, scope):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    feature_output = FeatureStageOutput(feature_matrix_path=tmp_path, feature_taxonomy_path=tmp_path,
                                         validation_report_path=tmp_path, identifier_col="customer_id",
                                         feature_cols=["avg_logins"])
    output = TrainingStageOutput(
        model_paths={"churned": model_path}, feature_importance_path=tmp_path, selected_features_path=tmp_path,
        train_report_path=tmp_path, selected_features={"churned": ["not_a_real_feature"]}, metric_values={"churned": 0.4},
    )
    with pytest.raises(ContractViolation, match="not present in the Feature stage's feature_cols"):
        verify_training_stage_output(output, feature_output, scope)


def test_verify_training_stage_output_passes_clean(tmp_path, scope):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    feature_output = FeatureStageOutput(feature_matrix_path=tmp_path, feature_taxonomy_path=tmp_path,
                                         validation_report_path=tmp_path, identifier_col="customer_id",
                                         feature_cols=["avg_logins"])
    output = TrainingStageOutput(
        model_paths={"churned": model_path}, feature_importance_path=tmp_path, selected_features_path=tmp_path,
        train_report_path=tmp_path, selected_features={"churned": ["avg_logins"]}, metric_values={"churned": 0.4},
    )
    verify_training_stage_output(output, feature_output, scope)  # no raise


# --- Validation stage --------------------------------------------------------

def test_verify_validation_stage_output_catches_inconsistent_pass_flag(scope):
    output = ValidationStageOutput(
        criteria_path="x", report_path="x",
        results={"churned": TargetValidationResult(metric_value=0.1, threshold=0.3, passed=True)},  # should be False
        passed=True,
    )
    with pytest.raises(ContractViolation, match="inconsistent with metric_value"):
        verify_validation_stage_output(output, scope)


def test_verify_validation_stage_output_passes_clean(scope):
    output = ValidationStageOutput(
        criteria_path="x", report_path="x",
        results={"churned": TargetValidationResult(metric_value=0.4, threshold=0.3, passed=True)},
        passed=True,
    )
    verify_validation_stage_output(output, scope)  # no raise
