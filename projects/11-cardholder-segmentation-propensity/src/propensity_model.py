"""
Response-propensity model: among customers who were actually sent the
growth team's past campaign, predict who responds. Trained on the
labeled subset (past_offer_sent == 1), then scored against the FULL
customer base, since the point is to rank everyone, offered or not,
for the next campaign.

At a ~32% response rate this isn't the extreme class imbalance projects
01/05/06 deal with, so ROC-AUC is a reasonable headline metric here,
not just PR-AUC. What matters more for a marketing use case is the
cumulative gains chart: given a fixed budget that can only reach some
fraction of customers, how many of the actual responders does the
model's ranking capture, against a random-targeting baseline.

Uses the same isotonic-calibration-on-a-validation-split pattern as
project 01, since expected-responder counts under a budget (computed
in targeting.py) need probabilities that are actually well-calibrated,
not just well-ranked.

Saves:
    reports/metrics.json
    reports/figures/{gains_chart, calibration_curve}.png
    reports/propensity_scores.csv (every customer, offered or not)
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

FEATURE_NUM = ["tenure_days", "lifetime_orders", "recency_days", "frequency_90d",
               "monetary_90d", "category_diversity", "decline_rate"]
FEATURE_CAT = ["segment", "primary_channel"]


def build_features(df):
    """One-hot encodes over the full customer base (not just the
    offered subset) so the resulting column set is identical whether
    this is called for training or for scoring the whole population
    later in targeting.py."""
    return pd.get_dummies(df[FEATURE_NUM + FEATURE_CAT], columns=FEATURE_CAT)


def cumulative_gains(y_true, y_score, n_bins=10):
    """Sorts by predicted score descending, splits into equal-size
    bins, and returns the cumulative share of population targeted vs.
    the cumulative share of actual positives captured at each bin edge,
    the standard marketing decile-gains view. Pure function: takes
    plain arrays, returns plain arrays, independently testable."""
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    n = len(y_sorted)
    total_positives = y_sorted.sum()

    cum_pop_share = [0.0]
    cum_capture_share = [0.0]
    for b in range(1, n_bins + 1):
        cutoff = int(round(n * b / n_bins))
        cum_pop_share.append(cutoff / n)
        cum_capture_share.append(y_sorted[:cutoff].sum() / total_positives if total_positives else 0.0)
    return np.array(cum_pop_share), np.array(cum_capture_share)


def main():
    customers = pd.read_csv(BASE / "data" / "customers.csv")
    segments = pd.read_csv(BASE / "reports" / "customer_segments.csv")
    df = customers.merge(segments[["customer_id", "segment"]], on="customer_id", how="left")

    X_full = build_features(df)
    offered_mask = df["past_offer_sent"] == 1

    X_offered = X_full[offered_mask].reset_index(drop=True)
    y_offered = df.loc[offered_mask, "responded"].astype(int).reset_index(drop=True)
    print(f"Offered customers: {len(X_offered):,} | response rate: {y_offered.mean():.1%}")

    # 60/20/20 train/val/test, stratified on the response label, since
    # there's no temporal dimension here (a single trailing-90-day
    # snapshot, unlike the time-ordered splits in 01/05/06).
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_offered, y_offered, test_size=0.4, stratify=y_offered, random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42,
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # --- Baseline: logistic regression ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    logit = LogisticRegression(max_iter=2000)
    logit.fit(X_train_s, y_train)
    logit_auc = roc_auc_score(y_test, logit.predict_proba(X_test_s)[:, 1])

    # --- Main model: gradient boosted trees, calibrated on the val split ---
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                          min_samples_leaf=40, random_state=42)
    gbm.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(gbm, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)

    y_prob_raw = gbm.predict_proba(X_test)[:, 1]
    y_prob_test = calibrated.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob_test)
    ap = average_precision_score(y_test, y_prob_test)
    brier_raw = brier_score_loss(y_test, y_prob_raw)
    brier_cal = brier_score_loss(y_test, y_prob_test)

    print(f"Logistic regression baseline AUC: {logit_auc:.4f}")
    print(f"GBM test ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f} (base rate {y_test.mean():.1%})")
    print(f"Brier score - raw: {brier_raw:.4f} | calibrated: {brier_cal:.4f}")

    SOURCE = f"Source: synthetic BNPL customer data · held-out test set · n = {len(X_test):,} offered customers"

    # --- Cumulative gains chart ---
    pop_share, capture_share = cumulative_gains(y_test.values, y_prob_test)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(pop_share * 100, capture_share * 100, color=SLATE, linewidth=1.8, marker="o", markersize=4,
            label="Model-ranked targeting")
    ax.plot([0, 100], [0, 100], "--", color=GREY, linewidth=1.1, label="Random targeting")
    top20_capture = np.interp(20, pop_share * 100, capture_share * 100)
    ax.plot([20], [top20_capture], marker="o", markersize=7, color=MUTED_RED, zorder=5)
    ax.annotate(f"Top 20% captures\n{top20_capture:.0f}% of responders", xy=(20, top20_capture),
                xytext=(32, top20_capture - 12), fontsize=9.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": 0.9})
    style_ax(ax, title="Cumulative gains: the model concentrates responders early",
             subtitle="Share of actual responders captured vs. share of customers targeted",
             xlabel="Share of customers targeted (%)", ylabel="Share of responders captured (%)",
             grid_axis="both")
    ax.legend(loc="lower right")
    savefig(fig, FIG_DIR / "gains_chart.png", footnote=SOURCE)

    # --- Calibration curve ---
    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=8, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob_test, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1.1, label="Perfect calibration")
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", markersize=4.5, color=MUTED_RED, linewidth=1.6, label="Raw GBM")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="o", markersize=4.5, color=SLATE, linewidth=1.6, label="Isotonic-calibrated")
    style_ax(ax, title="Calibration curve", subtitle="Predicted probability vs. observed response rate, by decile",
             xlabel="Mean predicted probability", ylabel="Observed response rate", grid_axis="both")
    ax.legend(loc="upper left")
    savefig(fig, FIG_DIR / "calibration_curve.png", footnote=SOURCE)

    # --- Score the FULL customer base, offered or not ---
    all_probs = calibrated.predict_proba(X_full)[:, 1]
    scores = df[["customer_id", "segment", "past_offer_sent"]].copy()
    scores["propensity_score"] = all_probs
    scores.to_csv(BASE / "reports" / "propensity_scores.csv", index=False)

    metrics = {
        "n_offered": int(len(X_offered)), "n_train": int(len(X_train)),
        "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "response_rate_offered": round(float(y_offered.mean()), 4),
        "logistic_baseline_auc": round(float(logit_auc), 4),
        "gbm_test_roc_auc": round(float(auc), 4),
        "gbm_test_pr_auc": round(float(ap), 4),
        "brier_raw": round(float(brier_raw), 4),
        "brier_calibrated": round(float(brier_cal), 4),
        "gains_top20pct_capture_share": round(float(top20_capture / 100), 4),
    }
    with open(BASE / "reports" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({
        "model": gbm, "calibrated_model": calibrated, "feature_columns": list(X_full.columns),
    }, BASE / "reports" / "model.pkl")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
