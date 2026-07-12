"""
Train and evaluate the delinquency model with a temporal split (mimics
real-world deployment: train on the past, test on the most recent months).

Saves:
    reports/metrics.json
    reports/figures/{roc_curve, pr_curve, calibration_curve, confusion_matrix,
                      threshold_cost_curve}.png
    reports/model.pkl (gitignored - regenerate via this script)
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler

from features import build_feature_pipeline, engineer_features, RAW_FEATURE_COLS
from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY, HEATMAP_CMAP, HEATMAP_TEXT_LOW, HEATMAP_TEXT_HIGH

BEST_PARAMS_DEFAULT = {
    "max_depth": 5, "learning_rate": 0.06, "max_iter": 300, "l2_regularization": 0.5,
}

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

# Business assumptions used to pick an operating threshold:
# - Approving a loan that goes delinquent costs (on average) ~70% of the
#   principal (net of recoveries/fees).
# - Declining a loan that would have been repaid forgoes the fee margin,
#   ~6% of principal.
COST_FALSE_NEGATIVE_RATE = 0.70   # missed a bad loan -> approved it
COST_FALSE_POSITIVE_RATE = 0.06   # declined a good loan -> lost margin


def temporal_split(df, train_end_month=18, val_end_month=21):
    train = df[df.origination_month <= train_end_month]
    val = df[(df.origination_month > train_end_month) & (df.origination_month <= val_end_month)]
    test = df[df.origination_month > val_end_month]
    return train, val, test


def find_cost_optimal_threshold(y_true, y_prob, loan_amounts):
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn_mask = (pred == 0) & (y_true == 1)
        fp_mask = (pred == 1) & (y_true == 0)
        cost = (loan_amounts[fn_mask] * COST_FALSE_NEGATIVE_RATE).sum() + \
               (loan_amounts[fp_mask] * COST_FALSE_POSITIVE_RATE).sum()
        costs.append(cost)
    costs = np.array(costs)
    best_idx = costs.argmin()
    return thresholds, costs, thresholds[best_idx]


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")
    df = engineer_features(df)
    train_df, val_df, test_df = temporal_split(df)
    print(f"Train: {len(train_df):,} (months 1-18) | Val: {len(val_df):,} (19-21) | "
          f"Test: {len(test_df):,} (22-24, incl. macro shock)")
    SOURCE = f"Source: synthetic BNPL loan data · held-out test set, months 22-24 · n = {len(test_df):,} loans"

    # Feature pipeline (missing-value imputation, outlier capping, one-hot
    # encoding) is fit on the training split only, then applied unchanged
    # to validation and test, the same statistics a frozen deployment
    # would use rather than ones re-computed on data the model shouldn't
    # get to see yet.
    feature_pipeline = build_feature_pipeline()
    X_train = feature_pipeline.fit_transform(train_df[RAW_FEATURE_COLS])
    X_val = feature_pipeline.transform(val_df[RAW_FEATURE_COLS])
    X_test = feature_pipeline.transform(test_df[RAW_FEATURE_COLS])
    feature_names = list(X_train.columns)
    y_train = train_df["delinquent_30dpd"].values
    y_val = val_df["delinquent_30dpd"].values
    y_test = test_df["delinquent_30dpd"].values

    # --- Baseline: logistic regression (interpretable benchmark) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(X_train_s, y_train)
    logit_auc = roc_auc_score(y_test, logit.predict_proba(X_test_s)[:, 1])

    # --- Main model: gradient boosted trees ---
    # Hyperparameters come from tune.py's time-series CV search (reports/best_params.json)
    # if it's been run; otherwise falls back to the original hand-picked defaults.
    best_params_path = BASE / "reports" / "best_params.json"
    if best_params_path.exists():
        with open(best_params_path) as f:
            tuned = json.load(f)
        gbm_params = tuned["best_params"]
        print(f"Using tuned hyperparameters from {tuned.get('n_cv_splits')}-fold time-series CV "
              f"(mean CV AUC {tuned['cv_mean_auc']:.3f} +/- {tuned['cv_std_auc']:.3f}): {gbm_params}")
    else:
        gbm_params = BEST_PARAMS_DEFAULT
        print("No reports/best_params.json found - run src/tune.py first for tuned hyperparameters. "
              "Falling back to defaults.")

    gbm = HistGradientBoostingClassifier(
        **gbm_params, random_state=42, class_weight="balanced",
    )
    gbm.fit(X_train, y_train)

    # --- Probability calibration on validation split ---
    calibrated = CalibratedClassifierCV(gbm, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)

    y_prob_test = calibrated.predict_proba(X_test)[:, 1]
    y_prob_raw = gbm.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob_test)
    ap = average_precision_score(y_test, y_prob_test)
    brier = brier_score_loss(y_test, y_prob_test)
    brier_raw = brier_score_loss(y_test, y_prob_raw)

    print(f"Logistic regression baseline AUC: {logit_auc:.3f}")
    print(f"GBM test AUC: {auc:.3f} | PR-AUC (AP): {ap:.3f}")
    print(f"Brier score - raw: {brier_raw:.4f} | calibrated: {brier:.4f}")

    # --- Cost-optimal threshold ---
    loan_amounts = test_df["loan_amount_usd"].values
    thresholds, costs, best_t = find_cost_optimal_threshold(y_test, y_prob_test, loan_amounts)
    default_cost = costs[np.argmin(np.abs(thresholds - 0.5))]
    best_cost = costs.min()
    print(f"Cost-optimal threshold: {best_t:.2f} "
          f"(portfolio cost ${best_cost:,.0f} vs ${default_cost:,.0f} at t=0.50, "
          f"{(1 - best_cost/default_cost):.1%} lower)")

    y_pred_best = (y_prob_test >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    # --- Figures ---
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=SLATE, linewidth=1.8, label=f"GBM (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1.1, label="Random")
    style_ax(ax, title="ROC curve",
             subtitle="Held-out months 22-24 (incl. shock)",
             xlabel="False positive rate", ylabel="True positive rate", grid_axis="both")
    ax.legend(loc="lower right")
    savefig(fig, FIG_DIR / "roc_curve.png", footnote=SOURCE)

    prec, rec, _ = precision_recall_curve(y_test, y_prob_test)
    base_rate = y_test.mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, color=SLATE, linewidth=1.8, label=f"GBM (AP={ap:.3f})")
    ax.axhline(base_rate, ls="--", color=GREY, linewidth=1.1, label=f"Base rate ({base_rate:.2%})")
    style_ax(ax, title="Precision-recall curve",
             subtitle="Held-out months 22-24 (incl. shock)",
             xlabel="Recall", ylabel="Precision", grid_axis="both")
    ax.legend(loc="upper right")
    savefig(fig, FIG_DIR / "pr_curve.png", footnote=SOURCE)

    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=10, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob_test, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1.1, label="Perfect calibration")
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", markersize=4.5, color=MUTED_RED, linewidth=1.6, label="Raw GBM")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="o", markersize=4.5, color=SLATE, linewidth=1.6, label="Isotonic-calibrated")
    style_ax(ax, title="Calibration curve",
             subtitle="Predicted probability vs. observed rate, by decile",
             xlabel="Mean predicted probability", ylabel="Observed delinquency rate", grid_axis="both")
    ax.legend(loc="upper left")
    savefig(fig, FIG_DIR / "calibration_curve.png", footnote=SOURCE)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(cm, cmap=HEATMAP_CMAP)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Current", "30+ DPD"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Current", "30+ DPD"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    style_ax(ax, title="Confusion matrix",
             subtitle=f"At the cost-optimal threshold ({best_t:.2f})",
             xlabel="Predicted", ylabel="Actual", grid_axis=None)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                     color=HEATMAP_TEXT_HIGH if cm[i, j] > cm.max() / 2 else HEATMAP_TEXT_LOW)
    savefig(fig, FIG_DIR / "confusion_matrix.png", footnote=SOURCE)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs / 1e6, color=SLATE, linewidth=1.8)
    ax.axvline(best_t, ls="--", color=MUTED_RED, linewidth=1.3, label=f"Cost-optimal t={best_t:.2f}")
    ax.axvline(0.50, ls=":", color=GREY, linewidth=1.3, label="Naive t=0.50")
    style_ax(ax, title="Expected portfolio cost by decision threshold",
             xlabel="Decision threshold", ylabel="Expected cost (USD, millions)", grid_axis="both")
    ax.legend()
    savefig(fig, FIG_DIR / "threshold_cost_curve.png", footnote=SOURCE)

    metrics = {
        "n_train": int(len(train_df)), "n_val": int(len(val_df)), "n_test": int(len(test_df)),
        "logistic_baseline_auc": round(float(logit_auc), 4),
        "gbm_test_auc": round(float(auc), 4),
        "gbm_test_pr_auc": round(float(ap), 4),
        "brier_raw": round(float(brier_raw), 4),
        "brier_calibrated": round(float(brier), 4),
        "cost_optimal_threshold": round(float(best_t), 3),
        "expected_cost_at_optimal_threshold_usd": round(float(best_cost), 2),
        "expected_cost_at_naive_threshold_usd": round(float(default_cost), 2),
        "cost_reduction_vs_naive": round(float(1 - best_cost / default_cost), 4),
        "precision_at_optimal_threshold": round(float(precision), 4),
        "recall_at_optimal_threshold": round(float(recall), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_base_rate": round(float(base_rate), 4),
        "gbm_params": gbm_params,
        "gbm_params_source": "tuned (time-series CV)" if best_params_path.exists() else "hand-picked default",
    }
    if best_params_path.exists():
        metrics["cv_mean_auc"] = tuned["cv_mean_auc"]
        metrics["cv_std_auc"] = tuned["cv_std_auc"]
        metrics["cv_n_splits"] = tuned["n_cv_splits"]
        metrics["cv_n_candidates_searched"] = tuned["n_candidates_searched"]
    with open(BASE / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({
        "model": gbm, "calibrated_model": calibrated, "feature_pipeline": feature_pipeline,
        "feature_names": feature_names, "threshold": best_t,
    }, BASE / "reports" / "model.pkl")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
