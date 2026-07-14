"""Stage ③ Feature Engineering. Turns the Data stage's training spine
+ a DS-authored `feature_spec.yaml` into a validated feature matrix:
joins each event source back to the spine via the manifest's join
graph, filters every event to `event_date <= decision_time` before
aggregating (the point-in-time guard in `guards.py`), one-hot encodes
categoricals, and caps the result at the Model Scope's `max_features`
by keeping the strongest signal first if the spec asks for more than
that.
"""
import json
from pathlib import Path

import pandas as pd

from bindings import FeatureSpec
from guards import assert_no_target_leakage, filter_point_in_time
from model_scope import ModelScope
from stage_io import DataStageOutput
from stage_io import FeatureStageOutput

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rank_by_signal(df: pd.DataFrame, candidate_cols: list[str], target_col: str) -> list[str]:
    """Rank candidate feature columns by |correlation| with the target,
    strongest first. Used only when the spec proposes more candidates
    than max_features allows."""
    corr = df[candidate_cols].corrwith(df[target_col]).abs().fillna(0)
    return corr.sort_values(ascending=False).index.tolist()


def run(scope: ModelScope, data_output: DataStageOutput, feature_spec: FeatureSpec, run_dir: Path) -> FeatureStageOutput:
    manifest = json.loads(data_output.manifest_path.read_text())
    identifier_col = manifest["identifier_col"]
    decision_time_col = manifest["decision_time_col"]

    training_data = pd.read_parquet(data_output.training_data_path)
    training_data[decision_time_col] = pd.to_datetime(training_data[decision_time_col])
    cutoffs = training_data.set_index(identifier_col)[decision_time_col]

    base = pd.read_csv(PROJECT_ROOT / manifest["base_population_path"])

    event_frames: dict[str, pd.DataFrame] = {}
    for name, source in manifest["event_sources"].items():
        raw = pd.read_csv(PROJECT_ROOT / source["path"])
        raw[source["event_date_col"]] = pd.to_datetime(raw[source["event_date_col"]])
        event_frames[name] = filter_point_in_time(raw, cutoffs, source["join_key"], source["event_date_col"])

    feature_matrix = training_data[[identifier_col] + [t.name for t in scope.targets]].copy()
    taxonomy: dict[str, list[str]] = {}
    engineered_cols: list[str] = []

    for idea in feature_spec.features:
        if idea.source == "base":
            values = base.set_index(identifier_col)[idea.column].reindex(feature_matrix[identifier_col]).reset_index(drop=True)
            if idea.agg != "passthrough":
                raise ValueError(f"feature '{idea.name}' has source='base' but agg='{idea.agg}', base columns must use agg='passthrough'")
            if values.dtype == object:
                dummies = pd.get_dummies(values, prefix=idea.name)
                for col in dummies.columns:
                    feature_matrix[col] = dummies[col].astype(int)
                    engineered_cols.append(col)
                    taxonomy.setdefault(idea.concept, []).append(col)
            else:
                feature_matrix[idea.name] = values.values
                engineered_cols.append(idea.name)
                taxonomy.setdefault(idea.concept, []).append(idea.name)
        else:
            source = manifest["event_sources"].get(idea.source)
            if source is None:
                raise ValueError(f"feature '{idea.name}' references source='{idea.source}', not in this run's bindings")
            events = event_frames[idea.source]
            grouped = events.groupby(source["join_key"])[idea.column]
            if idea.agg == "mean":
                agg_series = grouped.mean()
            elif idea.agg == "sum":
                agg_series = grouped.sum()
            elif idea.agg == "count":
                agg_series = grouped.count()
            else:
                raise ValueError(f"feature '{idea.name}' has source='{idea.source}' but agg='{idea.agg}', event-source features must aggregate")
            feature_matrix[idea.name] = agg_series.reindex(feature_matrix[identifier_col]).fillna(0).values
            engineered_cols.append(idea.name)
            taxonomy.setdefault(idea.concept, []).append(idea.name)

    target_names = [t.name for t in scope.targets]
    assert_no_target_leakage(engineered_cols, target_names)

    if len(engineered_cols) > scope.max_features:
        primary_target = scope.target(kind="classification").name if any(t.kind == "classification" for t in scope.targets) else scope.targets[0].name
        ranked = _rank_by_signal(feature_matrix, engineered_cols, primary_target)
        engineered_cols = ranked[: scope.max_features]
        taxonomy = {concept: [c for c in cols if c in engineered_cols] for concept, cols in taxonomy.items()}
        taxonomy = {concept: cols for concept, cols in taxonomy.items() if cols}

    feature_matrix = feature_matrix[[identifier_col] + target_names + engineered_cols]

    artifacts_dir = run_dir / "artifacts" / "features"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix_path = artifacts_dir / "features.parquet"
    feature_matrix.to_parquet(feature_matrix_path, index=False)

    taxonomy_lines = ["# Feature taxonomy", "", "Features grouped by analytical concept, not by source table.", ""]
    for concept, cols in sorted(taxonomy.items()):
        taxonomy_lines.append(f"## {concept}")
        for col in cols:
            taxonomy_lines.append(f"- `{col}`")
        taxonomy_lines.append("")
    feature_taxonomy_path = artifacts_dir / "feature_taxonomy.md"
    feature_taxonomy_path.write_text("\n".join(taxonomy_lines))

    null_rates = feature_matrix[engineered_cols].isna().mean()
    validation_lines = [
        "# Feature validation",
        "",
        f"- {len(engineered_cols)} features, cap was {scope.max_features}",
        f"- {len(feature_matrix):,} rows, one per `{identifier_col}`",
        f"- max null rate across features: {null_rates.max():.4f}" if len(null_rates) else "- no engineered features",
        f"- target leakage check: passed, none of {target_names} appear in the feature list",
    ]
    validation_report_path = artifacts_dir / "validation.md"
    validation_report_path.write_text("\n".join(validation_lines) + "\n")

    return FeatureStageOutput(
        feature_matrix_path=feature_matrix_path,
        feature_taxonomy_path=feature_taxonomy_path,
        validation_report_path=validation_report_path,
        identifier_col=identifier_col,
        feature_cols=engineered_cols,
    )
