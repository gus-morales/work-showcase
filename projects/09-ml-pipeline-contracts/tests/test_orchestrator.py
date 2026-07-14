"""End-to-end tests. Each test builds its own tiny run directory
against freshly generated synthetic data (not the committed
`runs/churn/run-001` example, which depends on `data/` being
regenerated first) so the pipeline is exercised as a whole without
depending on repo state outside of `tests/`.
"""
import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bindings import Bindings
from generate_data import generate_churn_domain
from model_scope import parse_model_scope_file
from orchestrator import run_pipeline
from stage_io import ContractViolation, FeatureStageOutput
from stages import data_prep, eda


def _write_churn_run(tmp_path: Path, n_customers: int = 400, threshold: float = 0.05) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    domain = generate_churn_domain(n_customers=n_customers, seed=3)
    domain["customers"].to_csv(data_dir / "customers.csv", index=False)
    domain["usage_events"].to_csv(data_dir / "usage_events.csv", index=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    bindings = {
        "identifier_col": "customer_id",
        "decision_time_col": "decision_date",
        "base_population": {"path": str(data_dir / "customers.csv")},
        "event_sources": {
            "usage_events": {
                "path": str(data_dir / "usage_events.csv"),
                "join_key": "customer_id",
                "event_date_col": "event_date",
            }
        },
    }
    (run_dir / "bindings.yaml").write_text(yaml.dump(bindings))

    feature_spec = {"features": [
        {"name": "avg_logins_pre_decision", "source": "usage_events", "column": "logins", "agg": "mean", "concept": "usage"},
        {"name": "tickets_opened_pre_decision", "source": "usage_events", "column": "support_tickets_opened", "agg": "sum", "concept": "usage"},
        {"name": "plan_tier", "source": "base", "column": "plan_tier", "agg": "passthrough", "concept": "segment"},
    ]}
    (run_dir / "feature_spec.yaml").write_text(yaml.dump(feature_spec))

    scope_md = textwrap.dedent(f"""\
        ---
        project:
          name: "Test churn"
          slug: test-churn
        objective: "test"
        problem: "test"
        task_type: classification
        targets:
          - name: churned_next_30d
            kind: classification
            raw_field: churned_next_30d
        metrics:
          - target: churned_next_30d
            name: pr_auc
            goal: maximize
            threshold: {threshold}
        max_features: 8
        data_sources: [usage_events]
        status: frozen
        ---

        test scope, no prose needed for this test.
        """)
    (run_dir / "MODEL_SCOPE.md").write_text(scope_md)
    return run_dir


def test_pipeline_runs_end_to_end_and_produces_every_declared_artifact(tmp_path):
    run_dir = _write_churn_run(tmp_path)
    result = run_pipeline(run_dir, verbose=False)

    assert result["eda"].sample_profile_path.exists()
    assert result["data"].training_data_path.exists()
    assert result["features"].feature_matrix_path.exists()
    for path in result["training"].model_paths.values():
        assert path.exists()
    assert result["validation"].report_path.exists()
    assert "churned_next_30d" in result["validation"].results


def test_pipeline_respects_max_features_cap(tmp_path):
    run_dir = _write_churn_run(tmp_path)
    result = run_pipeline(run_dir, verbose=False)
    assert len(result["features"].feature_cols) <= 8


def test_pipeline_refuses_to_run_against_a_draft_scope(tmp_path):
    run_dir = _write_churn_run(tmp_path)
    text = (run_dir / "MODEL_SCOPE.md").read_text().replace("status: frozen", "status: draft")
    (run_dir / "MODEL_SCOPE.md").write_text(text)

    with pytest.raises(ValueError, match="status='draft'"):
        run_pipeline(run_dir, verbose=False)


def test_pipeline_stops_at_the_feature_stage_when_the_contract_is_broken(tmp_path, monkeypatch):
    """The scenario `demo_broken_contract.py` shows in the README: a
    Feature stage that declares the target as one of its own features.
    The orchestrator has to stop there, before Training ever runs."""
    run_dir = _write_churn_run(tmp_path)

    def broken_feature_run(scope, data_output, feature_spec, run_dir):
        # Build a minimal, deliberately broken FeatureStageOutput: declares
        # the target itself as a feature column.
        df = pd.read_parquet(data_output.training_data_path)
        df["avg_logins_pre_decision"] = 1.0
        path = run_dir / "artifacts" / "features" / "features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return FeatureStageOutput(
            feature_matrix_path=path, feature_taxonomy_path=path, validation_report_path=path,
            identifier_col=data_output.identifier_col,
            feature_cols=["avg_logins_pre_decision", "churned_next_30d"],
        )

    monkeypatch.setattr("orchestrator.feature_engineering.run", broken_feature_run)

    with pytest.raises(ContractViolation, match="target leakage"):
        run_pipeline(run_dir, verbose=False)

    # Training must never have run: no model file was written.
    assert not (run_dir / "artifacts" / "model").exists()


def test_eda_and_data_stages_agree_on_row_count(tmp_path):
    run_dir = _write_churn_run(tmp_path, n_customers=250)
    scope = parse_model_scope_file(run_dir / "MODEL_SCOPE.md")
    bindings = Bindings(**yaml.safe_load((run_dir / "bindings.yaml").read_text()))

    eda_output = eda.run(scope, bindings, run_dir)
    data_output = data_prep.run(scope, bindings, run_dir)

    profile = json.loads(eda_output.sample_profile_path.read_text())
    assert profile["base_population"]["row_count"] == data_output.row_count == 250
