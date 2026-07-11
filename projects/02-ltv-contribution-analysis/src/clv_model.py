"""
Customer lifetime value modeling, two complementary approaches:

1. Probabilistic CLV (BG/NBD + Gamma-Gamma) - the standard non-contractual
   CLV approach. BG/NBD models purchase frequency and churn from each
   customer's recency/frequency/tenure; Gamma-Gamma models average order
   value conditional on frequency. Combined, they give a forward-looking
   expected value per customer without needing any features beyond
   transaction history. Validated with a calibration/holdout split.

2. Early-life predictive regression - a gradient boosting model that
   predicts a customer's 12-month revenue from only their first 30 days
   of behavior plus acquisition attributes. This is the model that would
   actually run in production, since it scores a customer immediately
   after signup rather than waiting for months of transaction history.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data, calibration_and_holdout_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_AMBER, GREY

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

OBS_END = pd.Timestamp("2026-01-01")
HOLDOUT_MONTHS = 6
HORIZON_MONTHS = 12


def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    return customers, orders


# ---------------------------------------------------------------------
# 1. Probabilistic CLV (BG/NBD + Gamma-Gamma)
# ---------------------------------------------------------------------

def fit_probabilistic_clv(customers, orders, source_note):
    summary = summary_data_from_transaction_data(
        orders, "customer_id", "order_date",
        monetary_value_col="order_value_mxn", observation_period_end=OBS_END, freq="D",
    )

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    repeat = summary[summary["frequency"] > 0].copy()
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])

    clv = ggf.customer_lifetime_value(
        bgf, summary["frequency"], summary["recency"], summary["T"],
        summary[summary["frequency"] > 0]["monetary_value"].reindex(summary.index, fill_value=summary["monetary_value"].mean()),
        time=HORIZON_MONTHS, freq="D", discount_rate=0.01,
    )
    clv = clv.rename("predicted_clv_12m_mxn")
    result = summary.join(clv)
    result.to_csv(BASE / "reports" / "customer_clv.csv")

    # --- CLV distribution chart ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vals = result["predicted_clv_12m_mxn"].clip(lower=0)
    ax.hist(vals, bins=60, color=SLATE, alpha=0.85, zorder=3)
    median = vals.median()
    ax.axvline(median, color=MUTED_AMBER, linewidth=1.5, linestyle="--")
    ax.text(median + vals.max() * 0.02, ax.get_ylim()[1] * 0.75, f"median\n${median:,.0f}",
            fontsize=9.5, color=MUTED_AMBER, va="top")
    style_ax(ax, title="Predicted 12-month value is concentrated in a small share of customers",
             subtitle="Distribution of BG/NBD + Gamma-Gamma predicted customer lifetime value",
             xlabel="Predicted 12-month CLV (MXN)", ylabel="Customers")
    savefig(fig, FIG_DIR / "clv_distribution.png", footnote=source_note)

    top_decile_share = vals.sort_values(ascending=False).head(int(len(vals) * 0.1)).sum() / vals.sum()
    print(f"Median predicted 12-month CLV: {median:,.0f} MXN")
    print(f"Top decile of customers captures {top_decile_share:.1%} of predicted 12-month value")
    return result, bgf, ggf


def validate_calibration_holdout(orders, source_note):
    calib_end = OBS_END - pd.DateOffset(months=HOLDOUT_MONTHS)
    cal_hold = calibration_and_holdout_data(
        orders, "customer_id", "order_date",
        calibration_period_end=calib_end, observation_period_end=OBS_END, freq="D",
    )
    cal_hold = cal_hold[cal_hold["frequency_cal"] >= 0]

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cal_hold["frequency_cal"], cal_hold["recency_cal"], cal_hold["T_cal"])

    holdout_days = (OBS_END - calib_end).days
    cal_hold["predicted_holdout"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        holdout_days, cal_hold["frequency_cal"], cal_hold["recency_cal"], cal_hold["T_cal"],
    )

    bucket_edges = [0, 1, 2, 3, 5, 8, cal_hold["frequency_cal"].max() + 1]
    cal_hold["freq_bucket"] = pd.cut(cal_hold["frequency_cal"], bins=bucket_edges, right=False)
    agg = cal_hold.groupby("freq_bucket", observed=True).agg(
        actual=("frequency_holdout", "mean"), predicted=("predicted_holdout", "mean"),
        n=("frequency_holdout", "size"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(agg))
    width = 0.35
    ax.bar(x - width / 2, agg["actual"], width, color=SLATE, label="Actual holdout purchases", zorder=3)
    ax.bar(x + width / 2, agg["predicted"], width, color=MUTED_TEAL, label="BG/NBD predicted", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in agg["freq_bucket"]], rotation=0, fontsize=9)
    style_ax(ax, title="BG/NBD holdout predictions track actual repeat purchases closely",
             subtitle=f"Customers grouped by calibration-period order count, {HOLDOUT_MONTHS}-month holdout",
             xlabel="Orders in calibration period", ylabel="Avg orders in holdout period")
    ax.legend(loc="upper left", fontsize=9.5)
    savefig(fig, FIG_DIR / "clv_calibration_holdout.png", footnote=source_note)

    mae = mean_absolute_error(cal_hold["frequency_holdout"], cal_hold["predicted_holdout"])
    print(f"Calibration/holdout MAE (orders): {mae:.3f}")


# ---------------------------------------------------------------------
# 2. Early-life predictive regression (day-30 features -> 12-month revenue)
# ---------------------------------------------------------------------

def build_early_life_dataset(customers, orders):
    # Only cohorts with a full 12 months of post-acquisition history observed
    eligible = customers[customers["cohort_month"] <= 24 - HORIZON_MONTHS].copy()
    merged = orders.merge(customers[["customer_id", "cohort_month"]], on="customer_id")
    merged["days_since_acq"] = merged["months_since_acquisition"] * 28  # 4-week months, matches generator

    day30 = merged[merged["days_since_acq"] <= 30]
    feats = day30.groupby("customer_id").agg(
        orders_first_30d=("order_id", "count"),
        revenue_first_30d=("order_value_mxn", "sum"),
    ).reindex(eligible["customer_id"]).fillna(0)

    second_order_day = (
        day30[day30["months_since_acquisition"] > 0]
        .groupby("customer_id")["days_since_acq"].min()
        .reindex(eligible["customer_id"])
    )
    feats["days_to_second_order"] = second_order_day.fillna(999)
    feats["reordered_in_30d"] = (second_order_day.notna()).astype(int)

    horizon = merged[merged["months_since_acquisition"] < HORIZON_MONTHS]
    target = horizon.groupby("customer_id")["order_value_mxn"].sum().reindex(eligible["customer_id"]).fillna(0)

    df = eligible.set_index("customer_id").join(feats).join(target.rename("revenue_12m"))
    df = df.reset_index()
    return df


def fit_early_life_model(df, source_note, n_customers):
    source_note = f"Source: synthetic BNPL transaction data, gradient boosting regression (day-30 features) · n = {n_customers:,} customers"
    cat_cols = ["acquisition_channel", "city_tier", "employment_type"]
    num_cols = ["orders_first_30d", "revenue_first_30d", "days_to_second_order", "reordered_in_30d"]
    X = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
    y = df["revenue_12m"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=7)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    print(f"Early-life model (holdout, n={len(X_test)}): R2={r2:.3f}, MAE={mae:,.0f} MXN")

    # --- Predicted vs actual ---
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(y_test, pred, s=14, color=SLATE, alpha=0.35, edgecolor="none", zorder=3)
    lims = [0, max(y_test.max(), pred.max()) * 1.02]
    ax.plot(lims, lims, color=GREY, linewidth=1.2, linestyle="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    style_ax(ax, title=f"Day-30 behavior is a meaningful but partial signal (R² = {r2:.2f})",
             subtitle="Holdout customers: predicted vs. actual 12-month revenue",
             xlabel="Actual 12-month revenue (MXN)", ylabel="Predicted 12-month revenue (MXN)")
    savefig(fig, FIG_DIR / "early_life_predicted_vs_actual.png", footnote=source_note)

    # --- Feature importance ---
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(importances.index, importances.values, color=SLATE, zorder=3)
    style_ax(ax, title="First-order timing and early spend drive the prediction most",
             subtitle="Gradient boosting feature importance, early-life 12-month revenue model",
             xlabel="Relative importance")
    savefig(fig, FIG_DIR / "early_life_feature_importance.png", footnote=source_note)

    return model, r2, mae


def main():
    customers, orders = load_data()
    n_customers = len(customers)
    source_note = f"Source: synthetic BNPL transaction data (lifetimes BG/NBD + Gamma-Gamma) · n = {n_customers:,} customers"

    fit_probabilistic_clv(customers, orders, source_note)
    validate_calibration_holdout(orders, source_note)

    df = build_early_life_dataset(customers, orders)
    df.to_csv(BASE / "reports" / "early_life_dataset.csv", index=False)
    fit_early_life_model(df, source_note, n_customers)

    print("Wrote reports/customer_clv.csv, early_life_dataset.csv, and 4 figures.")


if __name__ == "__main__":
    main()
