"""Stage ① EDA. Profiles the base population and every event source a
run's bindings point at, and reports the target's rate over time
(bucketed by decision time), the thing that later tells a human at
Model Scope sign-off whether the target is stable enough to model
as-is. Doesn't decide anything, doesn't touch the Model Scope; it only
produces the two artifacts the orchestrator hands to a human before
the Scope gets frozen.
"""
import json
from pathlib import Path

import pandas as pd

from bindings import Bindings
from model_scope import ModelScope
from stage_io import EDAOutput

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _column_profile(df: pd.DataFrame) -> dict:
    return {
        col: {
            "dtype": str(df[col].dtype),
            "null_rate": round(float(df[col].isna().mean()), 4),
            "n_unique": int(df[col].nunique()),
        }
        for col in df.columns
    }


def run(scope: ModelScope, bindings: Bindings, run_dir: Path) -> EDAOutput:
    base = pd.read_csv(PROJECT_ROOT / bindings.base_population.path)

    sample_profile = {
        "base_population": {
            "path": bindings.base_population.path,
            "row_count": len(base),
            "columns": _column_profile(base),
        },
        "event_sources": {},
    }
    for name, source in bindings.event_sources.items():
        events = pd.read_csv(PROJECT_ROOT / source.path)
        coverage = events[source.join_key].nunique() / len(base) if len(base) else 0.0
        sample_profile["event_sources"][name] = {
            "path": source.path,
            "row_count": len(events),
            "identifiers_covered": round(float(coverage), 4),
            "columns": _column_profile(events),
        }

    target_exploration = {}
    decision_time = pd.to_datetime(base[bindings.decision_time_col])
    month_bucket = decision_time.dt.to_period("M").astype(str)
    for target in scope.targets:
        raw = base[target.raw_field]
        entry = {"raw_field": target.raw_field, "kind": target.kind}
        if target.kind == "classification":
            entry["overall_rate"] = round(float(raw.mean()), 4)
            entry["rate_by_month"] = raw.groupby(month_bucket).mean().round(4).to_dict()
        else:
            entry["overall_mean"] = round(float(raw.mean()), 4)
            entry["overall_median"] = round(float(raw.median()), 4)
            entry["mean_by_month"] = raw.groupby(month_bucket).mean().round(4).to_dict()
        target_exploration[target.name] = entry

    artifacts_dir = run_dir / "artifacts" / "eda"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sample_profile_path = artifacts_dir / "sample_profile.json"
    target_exploration_path = artifacts_dir / "target_exploration.json"
    sample_profile_path.write_text(json.dumps(sample_profile, indent=2, default=str))
    target_exploration_path.write_text(json.dumps(target_exploration, indent=2, default=str))

    return EDAOutput(sample_profile_path=sample_profile_path, target_exploration_path=target_exploration_path)
