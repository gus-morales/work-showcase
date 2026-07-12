"""
Fair lending monitoring: disparate impact testing and adverse action
reason codes on the trained delinquency model's approve/decline
decisions.

`demographic_group` (see generate_data.py) is a synthetic stand-in for
a protected-class attribute under ECOA/Reg B. It is never used as a
model feature, matching real underwriting practice, but it is
generated with a mild dependence on city tier, which *is* a feature.
That mirrors how disparate impact can arise indirectly, through
facially-neutral variables correlated with protected-class membership,
even when the protected attribute itself is nowhere in the model. A
fair lending review has to test outcomes, since checking the feature
list alone would miss this.

Two checks:
1. Disparate impact: approval rate by group against the four-fifths
   rule (a group's approval rate should be at least 80% of the
   highest-approval group's), plus a two-proportion z-test for
   whether the gap is statistically distinguishable from noise.
2. Adverse action reason codes: for each declined applicant, the
   top SHAP-driven reasons, restricted to a fixed allowlist of
   genuinely credit-relevant factors. `city_tier`, `device_type`,
   `acquisition_channel`, and `merchant_category` are excluded from
   the reason-code universe by policy, even when they carry real SHAP
   signal, mirroring the standard practice of not citing geography or
   channel-of-acquisition on an adverse action notice.

Run:
    python src/fair_lending.py
Writes:
    reports/figures/fair_lending_approval_rate.png
    reports/figures/fair_lending_reason_codes.png
    reports/fair_lending_summary.json
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from statsmodels.stats.proportion import proportions_ztest

from features import engineer_features, RAW_FEATURE_COLS
from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY
from train import temporal_split

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

FOUR_FIFTHS_THRESHOLD = 0.80

# Ordered (prefix, label) pairs; first match wins. Longer/more specific
# prefixes are listed before shorter ones they'd otherwise be shadowed
# by (e.g. "credit_bureau_score_na" before "credit_bureau_score").
REASON_CODE_MAP = [
    ("credit_bureau_score_na", "Insufficient credit history on file"),
    ("credit_bureau_score", "Credit score"),
    ("low_bureau_score", "Credit score"),
    ("avg_prior_repayment_delay_days", "Delinquent past credit obligations"),
    ("num_active_loans_elsewhere", "Number of existing credit obligations"),
    ("monthly_income_usd", "Income insufficient for amount of credit requested"),
    ("loan_to_income_ratio", "Excessive obligations relative to income"),
    ("installment_to_income_ratio", "Excessive obligations relative to income"),
    ("installment_amount_usd", "Excessive obligations relative to income"),
    ("down_payment_ratio", "Down payment amount"),
    ("num_previous_loans", "Length of credit history with lender"),
    ("tenure_months_platform", "Length of relationship with lender"),
    ("is_new_customer", "Length of relationship with lender"),
    ("num_installments", "Term of credit requested"),
    ("loan_amount_usd", "Amount of credit requested"),
    ("age", "Insufficient credit references due to age"),
]
# Never surfaced as a reason code, even if they carry SHAP signal.
EXCLUDED_PREFIXES = ("city_tier_", "device_type_", "acquisition_channel_", "merchant_category_")


def reason_label(feature_name):
    for prefix, label in REASON_CODE_MAP:
        if feature_name.startswith(prefix):
            return label
    return None  # excluded or unmapped


def load_scored_test_set():
    """Reproduces train.py's exact split, pipeline, and threshold, then
    scores the held-out test set and attaches demographic_group (never
    passed to the model, only merged in afterward for the fairness
    check)."""
    df = pd.read_csv(BASE / "data" / "loans.csv")
    df = engineer_features(df)
    _, _, test_df = temporal_split(df)
    test_df = test_df.reset_index(drop=True)

    bundle = joblib.load(BASE / "reports" / "model.pkl")
    X_test = bundle["feature_pipeline"].transform(test_df[RAW_FEATURE_COLS])
    y_prob = bundle["calibrated_model"].predict_proba(X_test)[:, 1]
    threshold = bundle["threshold"]
    approved = y_prob < threshold  # predicted NOT delinquent -> approve

    test_df = test_df.copy()
    test_df["y_prob"] = y_prob
    test_df["approved"] = approved
    return test_df, bundle, X_test, threshold


def disparate_impact_test(test_df):
    rates = test_df.groupby("demographic_group")["approved"].agg(["mean", "count"])
    rates = rates.rename(columns={"mean": "approval_rate", "count": "n"})
    reference_group = rates["approval_rate"].idxmax()
    reference_rate = rates.loc[reference_group, "approval_rate"]
    rates["disparate_impact_ratio"] = rates["approval_rate"] / reference_rate

    results = {}
    for group, row in rates.iterrows():
        if group == reference_group:
            continue
        approvals = [
            int(round(row["approval_rate"] * row["n"])),
            int(round(reference_rate * rates.loc[reference_group, "n"])),
        ]
        ns = [int(row["n"]), int(rates.loc[reference_group, "n"])]
        z_stat, p_value = proportions_ztest(approvals, ns)
        results[group] = {
            "approval_rate": float(row["approval_rate"]),
            "disparate_impact_ratio": float(row["disparate_impact_ratio"]),
            "passes_four_fifths_rule": bool(row["disparate_impact_ratio"] >= FOUR_FIFTHS_THRESHOLD),
            "z_stat": float(z_stat),
            "p_value": float(p_value),
            "statistically_significant_gap": bool(p_value < 0.05),
        }
    return rates, reference_group, results


def top_reasons_from_shap_row(row_values, labels, top_n=3):
    """Pure selection logic: given one declined applicant's SHAP values
    (already restricted to the allowed reason-code columns) and their
    labels, return up to top_n distinct reason labels, ranked by SHAP
    value descending, keeping only reasons that actually pushed risk up
    (positive SHAP value) rather than every column regardless of sign."""
    order = np.argsort(row_values)[::-1]
    picked = []
    for i in order:
        if row_values[i] <= 0:
            break
        lab = labels[i]
        if lab not in picked:
            picked.append(lab)
        if len(picked) == top_n:
            break
    return picked


def adverse_action_reason_codes(test_df, bundle, X_test, top_n=3):
    declined_mask = ~test_df["approved"].values
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer(X_test)

    feature_names = bundle["feature_names"]
    labels = [reason_label(f) for f in feature_names]
    allowed_idx = [i for i, lab in enumerate(labels) if lab is not None]

    declined_values = shap_values.values[declined_mask][:, allowed_idx]
    declined_labels = [labels[i] for i in allowed_idx]

    top_reasons = [top_reasons_from_shap_row(row, declined_labels, top_n) for row in declined_values]

    top1_counts = pd.Series([r[0] for r in top_reasons if r]).value_counts()
    n_declined = int(declined_mask.sum())
    return top_reasons, top1_counts, n_declined


def approval_rate_chart(rates, reference_group, source_note):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    groups = list(rates.index)
    vals = rates["approval_rate"].values * 100
    colors = [SLATE if g == reference_group else MUTED_RED for g in groups]
    ax.bar(groups, vals, color=colors, width=0.5, zorder=3)
    four_fifths_line = rates.loc[reference_group, "approval_rate"] * FOUR_FIFTHS_THRESHOLD * 100
    ax.axhline(four_fifths_line, color=GREY, linewidth=1.3, linestyle="--",
               label=f"Four-fifths threshold ({four_fifths_line:.1f}%)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10, color="#333")
    style_ax(ax, title="Approval rate by demographic group vs. the four-fifths rule",
             subtitle="Held-out test set, at the cost-optimal decision threshold",
             ylabel="Approval rate (%)")
    ax.legend(fontsize=9, loc="lower right")
    savefig(fig, FIG_DIR / "fair_lending_approval_rate.png", footnote=source_note)


def reason_code_chart(top1_counts, n_declined, source_note):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts = top1_counts.sort_values(ascending=True)
    ax.barh(counts.index, counts.values / n_declined * 100, color=SLATE, zorder=3)
    for i, v in enumerate(counts.values):
        ax.text(v / n_declined * 100 + 0.5, i, f"{v/n_declined:.0%}", va="center", fontsize=9.5, color="#333")
    style_ax(ax, title="Primary adverse action reason among declined applicants",
             subtitle="Top SHAP-driven reason per decline, restricted to the allowed reason-code list",
             xlabel="Share of declines citing this as the primary reason (%)")
    savefig(fig, FIG_DIR / "fair_lending_reason_codes.png", footnote=source_note)


def main():
    test_df, bundle, X_test, threshold = load_scored_test_set()
    source_note = f"Source: synthetic BNPL loan data · held-out test set, months 22-24 · n = {len(test_df):,} loans"

    rates, reference_group, di_results = disparate_impact_test(test_df)
    print(f"Approval rates at threshold {threshold:.2f}:")
    print(rates[["approval_rate", "n", "disparate_impact_ratio"]].round(4))
    for group, r in di_results.items():
        verdict = "PASSES" if r["passes_four_fifths_rule"] else "FAILS"
        sig = "statistically significant" if r["statistically_significant_gap"] else "not statistically significant"
        print(f"{group} vs. {reference_group}: disparate impact ratio = {r['disparate_impact_ratio']:.3f}, "
              f"{verdict} the four-fifths rule (p={r['p_value']:.4f}, {sig})")

    top_reasons, top1_counts, n_declined = adverse_action_reason_codes(test_df, bundle, X_test)
    print(f"\n{n_declined:,} declined applicants. Primary reason breakdown:")
    print((top1_counts / n_declined).round(3))

    approval_rate_chart(rates, reference_group, source_note)
    reason_code_chart(top1_counts, n_declined, source_note)

    summary = {
        "reference_group": reference_group,
        "approval_rates": rates["approval_rate"].round(4).to_dict(),
        "disparate_impact_results": di_results,
        "n_declined": n_declined,
        "primary_reason_breakdown": (top1_counts / n_declined).round(4).to_dict(),
    }
    with open(BASE / "reports" / "fair_lending_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote reports/figures/fair_lending_approval_rate.png, "
          "reports/figures/fair_lending_reason_codes.png, reports/fair_lending_summary.json")


if __name__ == "__main__":
    main()
