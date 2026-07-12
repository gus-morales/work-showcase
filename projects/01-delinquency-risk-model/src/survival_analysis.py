"""
Time-to-default view of the same delinquency outcome: instead of asking
"will this loan go 30+ days past due within the observation window,"
this asks "how fast." Kaplan-Meier for the unconditional and
per-segment time-to-default curves, a Cox proportional hazards model
for which factors speed up or slow down time-to-default and by how
much (hazard ratios), and a concordance index as a survival-specific
read on discrimination, next to the classifier's AUC.

time_to_30dpd_days and event_observed (see generate_data.py) are
generated from the same systematic risk score as delinquent_30dpd, so
this is a different lens on the same underlying risk, not a separate
dataset. event_observed == 1 means the loan actually went 30+ days
past due inside the observation window; event_observed == 0 means it
was censored (still current when the window closed).

Run:
    python src/survival_analysis.py
Writes:
    reports/figures/survival_km_by_employment.png
    reports/figures/survival_hazard_ratios.png
    reports/survival_summary.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from features import engineer_features
from style import set_style, style_ax, savefig, PALETTE, SLATE, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

EMPLOYMENT_TYPES = ["salaried", "self_employed", "gig_economy", "informal"]

COX_NUMERIC_COLS = [
    "avg_prior_repayment_delay_days", "num_active_loans_elsewhere",
    "monthly_income_usd", "loan_amount_usd", "num_installments",
    "down_payment_ratio", "tenure_months_platform",
]
# Hazard ratios are easier to read on comparable scales; income and loan
# amount are rescaled to $1k units so a one-unit change means something.
RESCALE = {"monthly_income_usd": 1_000, "loan_amount_usd": 1_000}


def load_survival_frame():
    df = pd.read_csv(BASE / "data" / "loans.csv")
    df = engineer_features(df)
    df["credit_bureau_score_filled"] = df["credit_bureau_score"].fillna(df["credit_bureau_score"].median())
    df["bureau_score_missing"] = df["credit_bureau_score"].isna().astype(int)
    return df


def build_cox_covariates(df):
    cov = pd.DataFrame(index=df.index)
    for col in COX_NUMERIC_COLS:
        cov[col] = df[col] / RESCALE.get(col, 1)
    cov["credit_bureau_score_filled"] = df["credit_bureau_score_filled"] / 100  # per 100 points
    cov["bureau_score_missing"] = df["bureau_score_missing"]
    # Reference level: salaried.
    for et in ["self_employed", "gig_economy", "informal"]:
        cov[f"employment_{et}"] = (df["employment_type"] == et).astype(int)
    # Reference level: everything other than electronics/travel.
    for mc in ["electronics", "travel"]:
        cov[f"merchant_{mc}"] = (df["merchant_category"] == mc).astype(int)
    cov["time_to_30dpd_days"] = df["time_to_30dpd_days"]
    cov["event_observed"] = df["event_observed"]
    return cov


def fit_cox_model(cov):
    cph = CoxPHFitter()
    cph.fit(cov, duration_col="time_to_30dpd_days", event_col="event_observed")
    return cph


def km_by_employment(df):
    kmf = KaplanMeierFitter()
    medians = {}
    curves = {}
    for et in EMPLOYMENT_TYPES:
        mask = df["employment_type"] == et
        kmf.fit(df.loc[mask, "time_to_30dpd_days"], df.loc[mask, "event_observed"], label=et)
        medians[et] = kmf.median_survival_time_
        curves[et] = kmf.survival_function_
    logrank = multivariate_logrank_test(
        df["time_to_30dpd_days"], df["employment_type"], df["event_observed"]
    )
    return curves, medians, logrank


def km_chart(curves, source_note):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for et, color in zip(EMPLOYMENT_TYPES, PALETTE):
        sf = curves[et]
        ax.step(sf.index, sf.iloc[:, 0] * 100, where="post", label=et, color=color, linewidth=1.8)
    style_ax(ax, title="Time-to-default by employment type",
             subtitle="Kaplan-Meier estimate, share of loans not yet 30+ days past due",
             xlabel="Days since origination", ylabel="Share still current (%)")
    ax.legend(fontsize=9.5, loc="lower left")
    savefig(fig, FIG_DIR / "survival_km_by_employment.png", footnote=source_note)


def hazard_ratio_chart(cph, source_note):
    summary = cph.summary.copy()
    summary = summary.sort_values("exp(coef)")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(summary))
    hr = summary["exp(coef)"].values
    lower = summary["exp(coef) lower 95%"].values
    upper = summary["exp(coef) upper 95%"].values
    colors = [MUTED_RED if h > 1 else SLATE for h in hr]
    ax.hlines(y, lower, upper, color=GREY, linewidth=1.3, zorder=2)
    ax.scatter(hr, y, color=colors, s=45, zorder=3)
    ax.axvline(1.0, color=GREY, linewidth=1.1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xscale("log")
    style_ax(ax, title="What speeds up or slows down time-to-default",
             subtitle="Cox proportional hazards ratios with 95% confidence intervals (log scale)",
             xlabel="Hazard ratio (> 1 speeds up default, < 1 slows it down)")
    savefig(fig, FIG_DIR / "survival_hazard_ratios.png", footnote=source_note)


def main():
    df = load_survival_frame()
    source_note = f"Source: synthetic BNPL loan data · full portfolio · n = {len(df):,} loans"

    curves, medians, logrank = km_by_employment(df)
    print("Median time-to-default by employment type (NaN = median survival not reached):")
    for et, m in medians.items():
        print(f"  {et}: {m}")
    print(f"Log-rank test across employment types: p = {logrank.p_value:.2e}")

    cov = build_cox_covariates(df)
    cph = fit_cox_model(cov)
    print("\nCox model summary:")
    print(cph.summary[["coef", "exp(coef)", "p"]].round(4))
    print(f"\nConcordance index: {cph.concordance_index_:.4f}")

    km_chart(curves, source_note)
    hazard_ratio_chart(cph, source_note)

    summary = {
        "median_time_to_default_days_by_employment": {
            et: (None if pd.isna(m) else float(m)) for et, m in medians.items()
        },
        "logrank_p_value": float(logrank.p_value),
        "cox_concordance_index": float(cph.concordance_index_),
        "cox_hazard_ratios": {
            idx: {
                "hazard_ratio": float(row["exp(coef)"]),
                "p_value": float(row["p"]),
                "ci_lower": float(row["exp(coef) lower 95%"]),
                "ci_upper": float(row["exp(coef) upper 95%"]),
            }
            for idx, row in cph.summary.iterrows()
        },
    }
    with open(BASE / "reports" / "survival_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote reports/figures/survival_km_by_employment.png, "
          "reports/figures/survival_hazard_ratios.png, reports/survival_summary.json")


if __name__ == "__main__":
    main()
