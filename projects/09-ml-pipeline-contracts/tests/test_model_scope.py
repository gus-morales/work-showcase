import pytest
from pydantic import ValidationError

from model_scope import ModelScope, require_frozen


def _base_kwargs(**overrides):
    kwargs = dict(
        project={"name": "Test", "slug": "test"},
        objective="obj",
        problem="prob",
        task_type="classification",
        targets=[{"name": "churned", "kind": "classification", "raw_field": "churned"}],
        metrics=[{"target": "churned", "name": "pr_auc", "goal": "maximize", "threshold": 0.3}],
        max_features=10,
        data_sources=["usage_events"],
        status="frozen",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_scope_parses():
    scope = ModelScope(**_base_kwargs())
    assert scope.target().name == "churned"
    assert scope.metric_for("churned").name == "pr_auc"


def test_max_features_must_be_positive():
    with pytest.raises(ValidationError, match="max_features must be >= 1"):
        ModelScope(**_base_kwargs(max_features=0))


def test_data_sources_required():
    with pytest.raises(ValidationError, match="data_sources must list at least one source"):
        ModelScope(**_base_kwargs(data_sources=[]))


def test_classification_task_type_requires_matching_target_kind():
    with pytest.raises(ValidationError, match="requires a target of the same kind"):
        ModelScope(**_base_kwargs(
            targets=[{"name": "hours", "kind": "regression", "raw_field": "hours"}],
            metrics=[{"target": "hours", "name": "mae", "goal": "minimize", "threshold": 1.0}],
        ))


def test_task_type_both_requires_exactly_two_targets_of_each_kind():
    with pytest.raises(ValidationError, match="requires exactly two targets"):
        ModelScope(**_base_kwargs(task_type="both"))


def test_task_type_both_accepts_one_classification_one_regression():
    scope = ModelScope(**_base_kwargs(
        task_type="both",
        targets=[
            {"name": "will_escalate", "kind": "classification", "raw_field": "will_escalate"},
            {"name": "resolution_hours", "kind": "regression", "raw_field": "resolution_hours"},
        ],
        metrics=[
            {"target": "will_escalate", "name": "pr_auc", "goal": "maximize", "threshold": 0.4},
            {"target": "resolution_hours", "name": "mae", "goal": "minimize", "threshold": 4.5},
        ],
    ))
    assert scope.target(kind="classification").name == "will_escalate"
    assert scope.target(kind="regression").name == "resolution_hours"


def test_task_type_both_rejects_two_of_the_same_kind():
    with pytest.raises(ValidationError, match="one classification and one regression target"):
        ModelScope(**_base_kwargs(
            task_type="both",
            targets=[
                {"name": "a", "kind": "classification", "raw_field": "a"},
                {"name": "b", "kind": "classification", "raw_field": "b"},
            ],
            metrics=[
                {"target": "a", "name": "pr_auc", "goal": "maximize", "threshold": 0.4},
                {"target": "b", "name": "pr_auc", "goal": "maximize", "threshold": 0.4},
            ],
        ))


def test_duplicate_target_names_rejected():
    with pytest.raises(ValidationError, match="target names must be unique"):
        ModelScope(**_base_kwargs(
            task_type="both",
            targets=[
                {"name": "same", "kind": "classification", "raw_field": "a"},
                {"name": "same", "kind": "regression", "raw_field": "b"},
            ],
            metrics=[
                {"target": "same", "name": "pr_auc", "goal": "maximize", "threshold": 0.4},
                {"target": "same", "name": "mae", "goal": "minimize", "threshold": 1.0},
            ],
        ))


def test_metric_must_reference_every_target_exactly_once():
    with pytest.raises(ValidationError, match="expected one metric per target"):
        ModelScope(**_base_kwargs(metrics=[]))


def test_metric_target_mismatch_rejected():
    with pytest.raises(ValidationError, match="must reference each target exactly once"):
        ModelScope(**_base_kwargs(
            metrics=[{"target": "someone_else", "name": "pr_auc", "goal": "maximize", "threshold": 0.3}],
        ))


def test_require_frozen_blocks_draft():
    scope = ModelScope(**_base_kwargs(status="draft"))
    with pytest.raises(ValueError, match="status='draft'"):
        require_frozen(scope)


def test_require_frozen_allows_frozen():
    scope = ModelScope(**_base_kwargs(status="frozen"))
    require_frozen(scope)  # no raise
