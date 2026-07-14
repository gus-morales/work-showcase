"""Stage ④ Training. Fits one model per Model Scope target, runs a
5-step iterative feature reduction against the holdout split (drop the
weakest feature, refit, keep the step with the best holdout metric),
and reports IV (classification targets only) and PSI (every feature,
train vs. holdout split) alongside gain-based importance. PSI here is
a different check than project 01's: it's asking whether a feature
looks the same in the split that trained the model as in the split
that evaluated it, not whether production has drifted from a training
reference period. A feature that fails this PSI check would fail
before the model ever ships, project 01's is a live, ongoing check on
a model that already has.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from model_scope import ModelScope, TargetSpec
from stage_io import FeatureStageOutput, TrainingStageOutput

MIN_FEATURES = 3
MAX_REDUCTION_STEPS = 5


def _time_split(feature_matrix: pd.DataFrame, manifest: dict, project_root: Path, test_frac: float = 0.25):
    identifier_col = manifest["identifier_col"]
    training_data = pd.read_parquet(project_root / manifest["training_data_path"])
    decision_time = pd.to_datetime(training_data[manifest["decision_time_col"]])
    order = training_data.assign(_dt=decision_time).sort_values("_dt")[identifier_col]
    cutoff = int(len(order) * (1 - test_frac))
    train_ids = set(order.iloc[:cutoff])
    is_train = feature_matrix[identifier_col].isin(train_ids)
    return feature_matrix[is_train].reset_index(drop=True), feature_matrix[~is_train].reset_index(drop=True)


def _fit(kind: str, X: pd.DataFrame, y: pd.Series):
    model = GradientBoostingClassifier(random_state=9, n_estimators=120, max_depth=3) if kind == "classification" \
        else GradientBoostingRegressor(random_state=9, n_estimators=120, max_depth=3)
    model.fit(X, y)
    return model


def _score(kind: str, metric_name: str, model, X: pd.DataFrame, y: pd.Series) -> float:
    if kind == "classification":
        proba = model.predict_proba(X)[:, 1]
        if metric_name == "pr_auc":
            return float(average_precision_score(y, proba))
        if metric_name == "roc_auc":
            return float(roc_auc_score(y, proba))
        raise ValueError(f"unknown classification metric '{metric_name}'")
    pred = model.predict(X)
    if metric_name == "mae":
        return float(mean_absolute_error(y, pred))
    raise ValueError(f"unknown regression metric '{metric_name}'")


def _reduce_features(kind: str, metric_name: str, goal: str, X_train, y_train, X_test, y_test):
    features = list(X_train.columns)
    history = []
    for step in range(MAX_REDUCTION_STEPS):
        model = _fit(kind, X_train[features], y_train)
        score = _score(kind, metric_name, model, X_test[features], y_test)
        history.append({"step": step, "n_features": len(features), "features": list(features), "metric": round(score, 5)})
        if len(features) <= MIN_FEATURES:
            break
        importances = pd.Series(model.feature_importances_, index=features)
        weakest = importances.idxmin()
        features = [f for f in features if f != weakest]

    best = max(history, key=lambda r: r["metric"]) if goal == "maximize" else min(history, key=lambda r: r["metric"])
    final_model = _fit(kind, X_train[best["features"]], y_train)
    return final_model, best, history


def _information_value(feature: pd.Series, target: pd.Series, bins: int = 5) -> float:
    """Standard credit-scoring IV: only meaningful against a binary
    target, so this is computed for the classification target's model
    only (see `feature_importance.csv`'s iv column being blank for the
    regression target, when one exists)."""
    if feature.nunique() <= 2:
        binned = feature
    else:
        try:
            binned = pd.qcut(feature, bins, duplicates="drop")
        except ValueError:
            binned = feature
    df = pd.DataFrame({"bin": binned, "target": target})
    grouped = df.groupby("bin", observed=True)["target"].agg(["sum", "count"])
    grouped["non_event"] = grouped["count"] - grouped["sum"]
    total_event, total_non_event = grouped["sum"].sum(), grouped["non_event"].sum()
    if total_event == 0 or total_non_event == 0:
        return 0.0
    event_rate = (grouped["sum"] + 0.5) / (total_event + 0.5 * len(grouped))
    non_event_rate = (grouped["non_event"] + 0.5) / (total_non_event + 0.5 * len(grouped))
    woe = np.log(event_rate / non_event_rate)
    return float(((event_rate - non_event_rate) * woe).sum())


def _psi(train_values: pd.Series, test_values: pd.Series, bins: int = 8) -> float:
    edges = np.unique(np.quantile(train_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    train_counts, _ = np.histogram(train_values, bins=edges)
    test_counts, _ = np.histogram(test_values, bins=edges)
    train_pct = np.clip(train_counts / max(train_counts.sum(), 1), 1e-4, None)
    test_pct = np.clip(test_counts / max(test_counts.sum(), 1), 1e-4, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def run(scope: ModelScope, feature_output: FeatureStageOutput, manifest_path: Path, project_root: Path, run_dir: Path) -> TrainingStageOutput:
    manifest = json.loads(manifest_path.read_text())
    feature_matrix = pd.read_parquet(feature_output.feature_matrix_path)
    train_df, test_df = _time_split(feature_matrix, manifest, project_root)

    artifacts_dir = run_dir / "artifacts" / "model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_paths, selected_features, metric_values = {}, {}, {}
    importance_rows, report_sections, reduction_history = [], [], {}
    classification_target: TargetSpec | None = next((t for t in scope.targets if t.kind == "classification"), None)

    for target in scope.targets:
        metric = scope.metric_for(target.name)
        X_train, y_train = train_df[feature_output.feature_cols], train_df[target.name]
        X_test, y_test = test_df[feature_output.feature_cols], test_df[target.name]

        model, best_step, history = _reduce_features(target.kind, metric.name, metric.goal, X_train, y_train, X_test, y_test)
        features = best_step["features"]
        reduction_history[target.name] = {"metric_name": metric.name, "steps": [
            {"n_features": h["n_features"], "metric": h["metric"]} for h in history
        ]}

        model_path = artifacts_dir / f"model_{target.name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        model_paths[target.name] = model_path
        selected_features[target.name] = features
        metric_values[target.name] = best_step["metric"]

        importances = pd.Series(model.feature_importances_, index=features)
        for feat in features:
            iv = _information_value(train_df[feat], train_df[classification_target.name], bins=5) \
                if classification_target is not None else None
            importance_rows.append({
                "target": target.name,
                "feature": feat,
                "gain": round(float(importances[feat]), 5),
                "iv": round(iv, 5) if iv is not None else None,
                "psi_train_vs_holdout": round(_psi(train_df[feat], test_df[feat]), 5),
            })

        report_sections.append(
            f"## {target.name} ({target.kind})\n\n"
            f"- metric: {metric.name} = {best_step['metric']:.4f} (goal: {metric.goal}, threshold: {metric.threshold})\n"
            f"- reduction funnel: " + " -> ".join(f"{h['n_features']} feat / {metric.name}={h['metric']:.4f}" for h in history) + "\n"
            f"- selected features: {', '.join(features)}\n"
        )

    feature_importance_path = artifacts_dir / "feature_importance.csv"
    pd.DataFrame(importance_rows).to_csv(feature_importance_path, index=False)

    selected_features_path = artifacts_dir / "selected_features.json"
    selected_features_path.write_text(json.dumps({k: v for k, v in selected_features.items()}, indent=2))

    (artifacts_dir / "reduction_history.json").write_text(json.dumps(reduction_history, indent=2))

    train_report_path = artifacts_dir / "train_report.md"
    train_report_path.write_text("# Training report\n\n" + "\n".join(report_sections))

    return TrainingStageOutput(
        model_paths=model_paths,
        feature_importance_path=feature_importance_path,
        selected_features_path=selected_features_path,
        train_report_path=train_report_path,
        selected_features=selected_features,
        metric_values=metric_values,
    )
