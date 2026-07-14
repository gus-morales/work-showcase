"""
Record-level validation contract for the decision log, in the same
spirit as the pull-request schema check that gates a real governance
repo: a record missing what its impact level requires shouldn't be
treated as valid. `DecisionRecord` is a Pydantic model of one row;
`validate_dataframe()` runs it over a whole DataFrame and returns the
row indices that fail, plus why.
"""
from datetime import date
from typing import Literal

import pandas as pd
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
Status = Literal["abandoned", "reverted", "closed"]
Outcome = Literal["keep", "iterate", "rollback"]


class DecisionRecord(BaseModel):
    decision_id: int
    artifact_type: ArtifactType
    domain_tag: DomainTag
    impact_level: ImpactLevel
    status: Status
    proposed_date: date
    abandoned: bool
    approval_lag_days: float | None = None
    approved_date: date | None = None
    shipped_date: date | None = None
    ship_check_required: bool
    ship_check_due: date | None = None
    ship_check_on_time: bool | None = None
    metric_check_due: date | None = None
    metric_check_on_time: bool | None = None
    outcome: Outcome | None = None

    @model_validator(mode="after")
    def _check_consistency(self):
        if self.abandoned:
            if self.status != "abandoned":
                raise ValueError("abandoned record must have status='abandoned'")
            if self.approved_date is not None or self.shipped_date is not None:
                raise ValueError("abandoned record cannot have an approved_date or shipped_date")
            return self

        if self.approved_date is None or self.shipped_date is None:
            raise ValueError("a decision that wasn't abandoned must have approved_date and shipped_date")

        # Medium/high impact decisions carry a ship check; low impact ones don't.
        expects_ship_check = self.impact_level in ("medium", "high")
        if expects_ship_check != self.ship_check_required:
            raise ValueError(
                f"impact_level={self.impact_level} requires ship_check_required={expects_ship_check}"
            )
        if self.ship_check_required and self.ship_check_due is None:
            raise ValueError("ship_check_required=True must have a ship_check_due date")

        # Every shipped decision carries a metric check.
        if self.metric_check_due is None:
            raise ValueError("a shipped decision must have a metric_check_due date")

        if self.status in ("closed", "reverted") and self.outcome is None:
            raise ValueError(f"status={self.status} must have an outcome")
        if self.status == "reverted" and self.outcome != "rollback":
            raise ValueError("status='reverted' must have outcome='rollback'")
        if self.status == "closed" and self.outcome == "rollback":
            raise ValueError("outcome='rollback' must have status='reverted', not 'closed'")

        return self


def validate_dataframe(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Validate every row; return (row_index, error_message) for each
    failure. An empty list means every record in df is valid."""
    errors = []
    records = df.to_dict(orient="records")
    for i, record in enumerate(records):
        clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        try:
            DecisionRecord(**clean)
        except Exception as exc:
            errors.append((i, str(exc)))
    return errors
