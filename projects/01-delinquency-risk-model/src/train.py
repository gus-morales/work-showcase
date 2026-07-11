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

from features import build_design_matrix, engineer_features

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

    X_all, feature_names = build_design_matrix(df)
    X_train, y_train = X_all.loc[train_df.index], train_df["delinquent_30dpd"].values
    X_val, y_val = X_all.loc[val_df.index], val_df["delinquent_30dpd"].values
    X_test, y_test = X_all.loc[test_df.index], test_df["delinquent_30dpd"].values

    # --- Baseline: logistic regression (interpretable benchmark) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(X_train_s, y_train)
    logit_auc = roc_auc_score(y_test, logit.predict_proba(X_test_s)[:, 1])

    # --- Main model: gradient boosted trees ---
    gbm = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.06, max_iter=300, l2_regularization=0.5,
        random_state=42, class_weight="balanced",
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
    loan_amounts = test_df["loan_amount_mxn"].values
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
    ax.plot(fpr, tpr, color="#1f3b57", label=f"GBM (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve - held-out months 22-24 (incl. shock)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_test, y_prob_test)
    base_rate = y_test.mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, color="#2f7d9e", label=f"GBM (AP={ap:.3f})")
    ax.axhline(base_rate, ls="--", color="grey", label=f"Base rate ({base_rate:.2%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve - held-out months 22-24 (incl. shock)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=10, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob_test, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", color="#d9544d", label="Raw GBM")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="o", color="#1f3b57", label="Isotonic-calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed delinquency rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Current", "30+ DPD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Current", "30+ DPD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ cost-optimal threshold ({best_t:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs / 1e6, color="#1f3b57")
    ax.axvline(best_t, ls="--", color="#d9544d", label=f"Cost-optimal t={best_t:.2f}")
    ax.axvline(0.50, ls=":", color="grey", label="Naive t=0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected portfolio cost (MXN, millions)")
    ax.set_title("Cost-optimal threshold selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_cost_curve.png", dpi=150)
    plt.close(fig)

    # --- Feature importance (permutation-free, from HGB) ---
    gbm_importances = getattr(gbm, "feature_importances_", None)

    metrics = {
        "n_train": int(len(train_df)), "n_val": int(len(val_df)), "n_test": int(len(test_df)),
        "logistic_baseline_auc": round(float(logit_auc), 4),
        "gbm_test_auc": round(float(auc), 4),
        "gbm_test_pr_auc": round(float(ap), 4),
        "brier_raw": round(float(brier_raw), 4),
        "brier_calibrated": round(float(brier), 4),
        "cost_optimal_threshold": round(float(best_t), 3),
        "expected_cost_at_optimal_threshold_mxn": round(float(best_cost), 2),
        "expected_cost_at_naive_threshold_mxn": round(float(default_cost), 2),
        "cost_reduction_vs_naive": round(float(1 - best_cost / default_cost), 4),
        "precision_at_optimal_threshold": round(float(precision), 4),
        "recall_at_optimal_threshold": round(float(recall), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_base_rate": round(float(base_rate), 4),
    }
    with open(BASE / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({
        "model": gbm, "calibrated_model": calibrated, "feature_names": feature_names,
        "threshold": best_t,
    }, BASE / "reports" / "model.pkl")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
