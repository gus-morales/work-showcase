"""Stage ② Data. Turns the frozen Model Scope + a run's bindings into
the training spine: one row per identifier, the decision-time cutoff,
and every declared target's raw field, nothing else. No feature
engineering happens here on purpose, that's the next stage's job; this
stage's only real judgment call is already made, in `bindings.yaml`.
Its job is to stage the raw data cleanly and write the manifest that
tells Feature Engineering how to join back to the event sources.
"""
import json
import os
from pathlib import Path

import pandas as pd

from bindings import Bindings
from model_scope import ModelScope, require_frozen
from stage_io import DataStageOutput

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(scope: ModelScope, bindings: Bindings, run_dir: Path) -> DataStageOutput:
    require_frozen(scope)

    base = pd.read_csv(PROJECT_ROOT / bindings.base_population.path)
    identifier_col = bindings.identifier_col

    if base[identifier_col].duplicated().any():
        raise ValueError(f"base population has duplicate {identifier_col} values, one row per example is required")

    cols = [identifier_col, bindings.decision_time_col]
    for target in scope.targets:
        if target.raw_field not in base.columns:
            raise ValueError(f"Model Scope target '{target.name}' declares raw_field='{target.raw_field}', not a column in the base population")
        cols.append(target.raw_field)

    training_data = base[cols].rename(columns={t.raw_field: t.name for t in scope.targets})

    artifacts_dir = run_dir / "artifacts" / "data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    training_data_path = artifacts_dir / "training_data.parquet"
    training_data.to_parquet(training_data_path, index=False)

    dictionary_lines = [
        "# Data dictionary",
        "",
        f"One row per `{identifier_col}`. `{bindings.decision_time_col}` is the point-in-time "
        "cutoff every downstream feature must respect.",
        "",
        "| Column | Source | Notes |",
        "|---|---|---|",
        f"| `{identifier_col}` | base population | primary identifier |",
        f"| `{bindings.decision_time_col}` | base population | point-in-time cutoff |",
    ]
    for target in scope.targets:
        dictionary_lines.append(f"| `{target.name}` | base population (`{target.raw_field}`) | target, kind={target.kind} |")
    data_dictionary_path = artifacts_dir / "data_dictionary.md"
    data_dictionary_path.write_text("\n".join(dictionary_lines) + "\n")

    manifest = {
        "identifier_col": identifier_col,
        "decision_time_col": bindings.decision_time_col,
        # relpath, not Path.relative_to: a run directory can live outside
        # PROJECT_ROOT (e.g. a test's tmp_path), and relpath still
        # produces something PROJECT_ROOT / ... resolves back correctly,
        # relative_to() would just raise in that case.
        "training_data_path": os.path.relpath(training_data_path, PROJECT_ROOT),
        "targets": [{"name": t.name, "kind": t.kind} for t in scope.targets],
        "event_sources": {
            name: {"path": source.path, "join_key": source.join_key, "event_date_col": source.event_date_col}
            for name, source in bindings.event_sources.items()
        },
        "base_population_path": bindings.base_population.path,
    }
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return DataStageOutput(
        training_data_path=training_data_path,
        data_dictionary_path=data_dictionary_path,
        manifest_path=manifest_path,
        identifier_col=identifier_col,
        row_count=len(training_data),
    )
