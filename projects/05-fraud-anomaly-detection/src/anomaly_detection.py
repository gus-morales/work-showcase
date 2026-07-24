"""
How much does having fraud labels actually buy you? Isolation Forest
scores each transaction's "unusualness" using only the feature values,
no labels at all, then that ranking is checked against the same
held-out fraud outcomes the supervised GBM in train.py was evaluated
against. This is the scenario a fraud team faces before enough
confirmed-fraud labels exist to train a supervised model, or for a
brand-new fraud pattern the existing labels don't cover.

Run:
    python src/anomaly_detection.py
Writes:
    reports/figures/anomaly_vs_supervised_pr_curve.png
    reports/anomaly_detection_summary.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve

from features import build_feature_pipeline, temporal_split, RAW_FEATURE_COLS
from style import set_style, style_ax, savefig, SLATE, MUTED_AMBER, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

ISOLATION_FOREST_PARAMS = {"n_estimators": 300, "contamination": "auto", "random_state": 42}


def fit_isolation_forest(X_train):
    """Isolation Forest never sees is_fraud; it isolates points that are
    easy to separate from the rest with few random splits, the
    unsupervised definition of "unusual." Trained on the same
    leakage-safe feature matrix the supervised model uses."""
    iso = IsolationForest(**ISOLATION_FOREST_PARAMS)
    iso.fit(X_train)
    return iso


def anomaly_scores(iso, X):
    """IsolationForest.score_samples is higher for more normal points;
    negate it so higher means more anomalous, matching the convention
    the supervised model's fraud probability already uses."""
    return -iso.score_samples(X)


def fit_supervised_gbm(X_train, y_train):
    gbm = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, max_iter=300, min_samples_leaf=40, random_state=42,
    )
    gbm.fit(X_train, y_train)
    return gbm


def comparison_pr_chart(y_test, supervised_prob, anomaly_score, source_note):
    base_rate = y_test.mean()
    prec_sup, rec_sup, _ = precision_recall_curve(y_test, supervised_prob)
    prec_iso, rec_iso, _ = precision_recall_curve(y_test, anomaly_score)
    ap_sup = average_precision_score(y_test, supervised_prob)
    ap_iso = average_precision_score(y_test, anomaly_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_sup, prec_sup, color=SLATE, linewidth=1.8, label=f"Supervised GBM (PR-AUC={ap_sup:.3f})")
    ax.plot(rec_iso, prec_iso, color=MUTED_AMBER, linewidth=1.8, label=f"Isolation Forest (PR-AUC={ap_iso:.3f})")
    ax.axhline(base_rate, ls="--", color=GREY, linewidth=1.1, label=f"Random ({base_rate:.2%})")
    style_ax(ax, title="Labels are worth a lot, but unsupervised still beats random",
             subtitle="Precision-recall, supervised model vs. unsupervised anomaly score, same held-out set",
             xlabel="Recall", ylabel="Precision", grid_axis="both")
    ax.legend(loc="upper right", fontsize=9)
    savefig(fig, FIG_DIR / "anomaly_vs_supervised_pr_curve.png", footnote=source_note)
    return ap_sup, ap_iso


def main():
    df = pd.read_csv(BASE / "data" / "transactions.csv", parse_dates=["timestamp"])
    train_df, val_df, test_df = temporal_split(df)
    source_note = f"Source: synthetic bank card-transaction data · held-out test set · n = {len(test_df):,} transactions"

    feature_pipeline = build_feature_pipeline()
    X_train = feature_pipeline.fit_transform(train_df[RAW_FEATURE_COLS])
    X_test = feature_pipeline.transform(test_df[RAW_FEATURE_COLS])
    y_train = train_df["is_fraud"].values
    y_test = test_df["is_fraud"].values

    gbm = fit_supervised_gbm(X_train, y_train)
    supervised_prob = gbm.predict_proba(X_test)[:, 1]

    iso = fit_isolation_forest(X_train)
    iso_score = anomaly_scores(iso, X_test)

    ap_sup, ap_iso = comparison_pr_chart(y_test, supervised_prob, iso_score, source_note)
    lift_sup = ap_sup / y_test.mean()
    lift_iso = ap_iso / y_test.mean()

    print(f"Supervised GBM: PR-AUC = {ap_sup:.4f} ({lift_sup:.1f}x the {y_test.mean():.2%} base rate)")
    print(f"Isolation Forest (unsupervised): PR-AUC = {ap_iso:.4f} ({lift_iso:.1f}x base rate)")
    print(f"Supervised advantage: {ap_sup / ap_iso:.1f}x the unsupervised PR-AUC")

    summary = {
        "test_base_rate": round(float(y_test.mean()), 5),
        "supervised_gbm_pr_auc": round(float(ap_sup), 4),
        "isolation_forest_pr_auc": round(float(ap_iso), 4),
        "supervised_lift_over_base_rate": round(float(lift_sup), 2),
        "isolation_forest_lift_over_base_rate": round(float(lift_iso), 2),
        "supervised_advantage_ratio": round(float(ap_sup / ap_iso), 2),
    }
    with open(BASE / "reports" / "anomaly_detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote reports/figures/anomaly_vs_supervised_pr_curve.png, reports/anomaly_detection_summary.json")


if __name__ == "__main__":
    main()
