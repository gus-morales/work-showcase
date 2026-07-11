"""
Simple production-monitoring simulation: compares the "reference" scoring
population (train months) against the most recent months (which include the
synthetic macro shock) using Population Stability Index (PSI), and checks
whether model performance degrades on the shocked population.

This mirrors the kind of model-monitoring / data-quality checks referenced
on the CV (feature store + model monitoring work).
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import build_design_matrix, engineer_features

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"

MONITORED_FEATURES = [
    "credit_bureau_score", "loan_to_income_ratio", "avg_prior_repayment_delay_days",
    "num_active_loans_elsewhere", "monthly_income_mxn", "down_payment_ratio",
]

PSI_ALERT_THRESHOLD = 0.20  # >0.2 is commonly treated as "significant shift"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(reference, quantiles))
    if len(cut_points) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=cut_points)
    cur_counts, _ = np.histogram(current, bins=cut_points)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")
    df = engineer_features(df)

    reference = df[df.origination_month <= 21]     # what the model was trained on
    monitored = df[df.origination_month > 21]       # months 22-24, incl. macro shock

    psi_results = {
        feat: round(psi(reference[feat].values, monitored[feat].values), 4)
        for feat in MONITORED_FEATURES
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    feats = list(psi_results.keys())
    vals = list(psi_results.values())
    colors = ["#d9544d" if v > PSI_ALERT_THRESHOLD else "#2f7d9e" for v in vals]
    ax.barh(feats, vals, color=colors)
    ax.axvline(PSI_ALERT_THRESHOLD, ls="--", color="grey", label=f"Alert threshold ({PSI_ALERT_THRESHOLD})")
    ax.set_xlabel("PSI (reference: months 1-21, current: months 22-24)")
    ax.set_title("Feature drift monitoring (PSI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drift_psi.png", dpi=150)
    plt.close(fig)

    # Performance degradation check: score the "current" window with the
    # frozen model and see how AUC compares to the original held-out test AUC.
    bundle = joblib.load(BASE / "reports" / "model.pkl")
    calibrated = bundle["calibrated_model"]
    X_all, _ = build_design_matrix(df)
    X_monitored = X_all.loc[monitored.index]
    y_monitored = monitored["delinquent_30dpd"].values
    y_prob = calibrated.predict_proba(X_monitored)[:, 1]
    auc_monitored = roc_auc_score(y_monitored, y_prob)

    with open(BASE / "reports" / "metrics.json") as f:
        base_metrics = json.load(f)
    auc_test = base_metrics["gbm_test_auc"]

    mean_pred_prob = float(y_prob.mean())
    observed_rate_monitored = float(monitored["delinquent_30dpd"].mean())

    report = {
        "psi_by_feature": psi_results,
        "features_flagged": [f for f, v in psi_results.items() if v > PSI_ALERT_THRESHOLD],
        "auc_original_test_month24": auc_test,
        "auc_monitored_window_22_24": round(float(auc_monitored), 4),
        "observed_rate_reference_months_1_21": round(float(reference["delinquent_30dpd"].mean()), 4),
        "observed_rate_monitored_months_22_24": round(observed_rate_monitored, 4),
        "mean_predicted_prob_monitored_window": round(mean_pred_prob, 4),
        "calibration_gap_monitored_window": round(observed_rate_monitored - mean_pred_prob, 4),
    }

    # Predicted vs. actual rate by month - shows the calibration break even
    # though input-feature PSI stayed clean (concept drift, not covariate drift).
    monitored_scored = monitored.copy()
    monitored_scored["pred_prob"] = y_prob
    by_month = monitored_scored.groupby("origination_month").agg(
        actual_rate=("delinquent_30dpd", "mean"), predicted_rate=("pred_prob", "mean"),
    )
    ref_by_month = reference.groupby("origination_month")["delinquent_30dpd"].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ref_by_month.index, ref_by_month.values * 100, marker="o", color="#57b8b0",
            label="Actual rate (training window)")
    ax.plot(by_month.index, by_month["actual_rate"] * 100, marker="o", color="#d9544d",
            label="Actual rate (monitored window)")
    ax.plot(by_month.index, by_month["predicted_rate"] * 100, marker="s", ls="--", color="#1f3b57",
            label="Model-predicted rate (monitored window)")
    ax.axvline(21.5, ls=":", color="grey", label="Train/monitor cutoff")
    ax.set_xlabel("Origination month")
    ax.set_ylabel("30+ DPD rate (%)")
    ax.set_title("Predicted vs. actual delinquency rate - calibration drift check")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "drift_predicted_vs_actual.png", dpi=150)
    plt.close(fig)
    with open(BASE / "reports" / "drift_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Drift monitoring report\n",
             f"Reference window: origination months 1-21. Monitored window: months 22-24 "
             f"(includes the synthetic macro-shock).\n",
             "\n## Population Stability Index by feature\n"]
    psi_df = pd.Series(psi_results, name="PSI").sort_values(ascending=False).to_frame()
    psi_df["flag"] = psi_df["PSI"].apply(lambda v: "ALERT" if v > PSI_ALERT_THRESHOLD else "ok")
    lines.append(psi_df.to_markdown())
    lines.append(f"\n\n## Delinquency rate shift\n- Reference: {report['observed_rate_reference_months_1_21']:.2%}"
                 f"\n- Monitored: {report['observed_rate_monitored_months_22_24']:.2%}\n")
    lines.append(f"\n## Model performance on monitored window\n- AUC (original held-out test set): "
                 f"{auc_test:.3f}\n- AUC (monitored window, months 22-24): {auc_monitored:.3f}\n")
    lines.append(f"\n## Calibration drift\n- Mean predicted probability (monitored window): "
                 f"{mean_pred_prob:.2%}\n- Observed delinquency rate (monitored window): "
                 f"{observed_rate_monitored:.2%}\n- Gap: {report['calibration_gap_monitored_window']:.2%}. "
                 f"The model under-predicts risk once the shock hits, even though PSI is clean. This is "
                 f"concept drift, not covariate drift. Outcome-rate monitoring catches it; PSI alone does not.\n")
    (BASE / "reports" / "drift_report.md").write_text("\n".join(lines))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
