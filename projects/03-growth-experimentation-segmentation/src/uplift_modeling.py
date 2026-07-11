"""
Uplift / CATE (conditional average treatment effect) modeling on the
repayment-reminder A/B test. The standard analysis (experiment_design.py)
reports one number: the *average* treatment effect across everyone. That
average can hide real heterogeneity, some segments might benefit far more
than others, which matters directly for rollout decisions (who actually
needs the redesign vs. who it's wasted effort on).

Approach: a T-learner (two gradient-boosted classifiers, one fit on each
arm, both predicting P(converted)) trained on a train split, then used to
score predicted individual treatment effects on a held-out test split.
Since true individual effects are never observable, the model is
validated the standard way for uplift models: by predicted-CATE decile
and with a Qini (cumulative gain) curve, both of which only use each
test user's own actual arm and actual outcome, not the model's
prediction for the arm they weren't in.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

RAW_COVARIATES = ["tenure_days", "sessions_pre_30d", "revenue_pre_30d_usd"]
OUTCOME = "converted_post_14d"
SEED = 13


def add_model_features(df):
    """log1p(tenure_days) instead of raw tenure_days: the true effect
    decays exponentially with tenure, and a tree-based model with a
    modest sample size per arm recovers that shape far more reliably
    from a log-scaled feature than from the raw (heavily right-skewed)
    day count."""
    df = df.copy()
    df["log_tenure_days"] = np.log1p(df["tenure_days"])
    return df


MODEL_COVARIATES = ["log_tenure_days", "sessions_pre_30d", "revenue_pre_30d_usd"]


def fit_t_learner(train_df):
    """Two independent classifiers, one per arm, both predicting
    P(converted) from the same covariates. Pure model-fitting, no
    plotting or I/O, so it's directly testable on a tiny fixture.
    Deliberately shallow and heavily leaf-regularized: the difference of
    two separately-fit models (the core T-learner mechanic) amplifies
    whatever noise each one picks up, so an under-regularized model here
    would mostly be estimating noise rather than the treatment effect."""
    treat = train_df[train_df.arm == "treatment"]
    control = train_df[train_df.arm == "control"]
    model_treat = HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3, max_iter=80, min_samples_leaf=120,
    )
    model_control = HistGradientBoostingClassifier(
        random_state=SEED, max_depth=3, max_iter=80, min_samples_leaf=120,
    )
    model_treat.fit(treat[MODEL_COVARIATES], treat[OUTCOME])
    model_control.fit(control[MODEL_COVARIATES], control[OUTCOME])
    return model_treat, model_control


def predict_cate(model_treat, model_control, X):
    p_treat = model_treat.predict_proba(X[MODEL_COVARIATES])[:, 1]
    p_control = model_control.predict_proba(X[MODEL_COVARIATES])[:, 1]
    return p_treat - p_control


def compute_uplift_by_decile(cate_pred, arm, outcome, n_buckets=5):
    """Pure computation. Buckets test users by predicted CATE, then
    computes each bucket's *realized* lift (actual outcome, actual arm,
    no model prediction involved in the realized number) so predicted
    and realized can be compared honestly. Highest predicted CATE first."""
    df = pd.DataFrame({"cate_pred": np.asarray(cate_pred), "arm": np.asarray(arm), "outcome": np.asarray(outcome)})
    df["bucket"] = pd.qcut(df["cate_pred"], n_buckets, labels=False, duplicates="drop")

    rows = []
    for b, g in df.groupby("bucket"):
        treat = g[g.arm == "treatment"]
        control = g[g.arm == "control"]
        realized_lift = (
            treat["outcome"].mean() - control["outcome"].mean()
            if len(treat) > 0 and len(control) > 0 else np.nan
        )
        rows.append({
            "bucket": int(b), "n": len(g),
            "predicted_cate_mean": float(g["cate_pred"].mean()),
            "realized_lift": float(realized_lift) if pd.notna(realized_lift) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("bucket", ascending=False).reset_index(drop=True)


def compute_qini_curve(cate_pred, arm, outcome):
    """Pure computation of the Qini (cumulative gain) curve: sort by
    predicted CATE descending, and at each population fraction compare
    cumulative treated conversions against cumulative control conversions
    scaled up to the same group size. The Qini coefficient is the area
    between this curve and the random-targeting diagonal."""
    df = pd.DataFrame({
        "cate_pred": np.asarray(cate_pred), "arm": np.asarray(arm), "outcome": np.asarray(outcome),
    }).sort_values("cate_pred", ascending=False).reset_index(drop=True)
    n = len(df)

    is_treat = (df["arm"] == "treatment").astype(int).values
    is_control = 1 - is_treat
    y = df["outcome"].values

    cum_treat_n = np.cumsum(is_treat)
    cum_control_n = np.cumsum(is_control)
    cum_treat_y = np.cumsum(y * is_treat)
    cum_control_y = np.cumsum(y * is_control)

    ratio = np.divide(cum_treat_n, cum_control_n, out=np.zeros_like(cum_treat_n, dtype=float), where=cum_control_n != 0)
    qini_values = cum_treat_y - cum_control_y * ratio
    population_fraction = np.arange(1, n + 1) / n

    total_gain = qini_values[-1]
    random_baseline = population_fraction * total_gain

    frac_full = np.concatenate(([0.0], population_fraction))
    qini_full = np.concatenate(([0.0], qini_values))
    baseline_full = np.concatenate(([0.0], random_baseline))
    qini_coefficient = float(np.trapezoid(qini_full - baseline_full, frac_full))

    return {
        "population_fraction": population_fraction, "qini_values": qini_values,
        "random_baseline": random_baseline, "qini_coefficient": qini_coefficient,
    }


def uplift_modeling(df, source_note):
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=SEED, stratify=df["arm"])
    train_df = add_model_features(train_df)
    test_df = add_model_features(test_df)

    model_treat, model_control = fit_t_learner(train_df)
    test_df["cate_pred"] = predict_cate(model_treat, model_control, test_df)

    # --- Bucket calibration: does a higher predicted CATE actually mean a bigger realized lift? ---
    bucket_df = compute_uplift_by_decile(test_df["cate_pred"], test_df["arm"], test_df[OUTCOME])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(bucket_df))
    width = 0.35
    ax.bar(x - width / 2, bucket_df["predicted_cate_mean"] * 100, width, color=GREY, label="Predicted CATE", zorder=3)
    ax.bar(x + width / 2, bucket_df["realized_lift"] * 100, width, color=SLATE, label="Realized lift (held-out)", zorder=3)
    ax.axhline(0, color=GREY, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in range(len(bucket_df))])
    style_ax(ax, title="Predicted CATE tracks the realized lift across quintiles",
             subtitle="Test-set users bucketed by predicted CATE, quintile 1 = highest predicted uplift",
             xlabel="Predicted-CATE quintile (highest to lowest)", ylabel="Conversion lift (pp)")
    ax.legend(fontsize=9.5, loc="upper right")
    savefig(fig, FIG_DIR / "uplift_calibration.png", footnote=source_note)

    # --- Qini curve: cumulative gain from targeting by predicted CATE vs. random targeting ---
    qini = compute_qini_curve(test_df["cate_pred"], test_df["arm"], test_df[OUTCOME])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(qini["population_fraction"] * 100, qini["qini_values"], color=SLATE, linewidth=1.8, label="Targeting by predicted CATE")
    ax.plot(qini["population_fraction"] * 100, qini["random_baseline"], color=GREY, linewidth=1.4, ls="--", label="Random targeting")
    style_ax(ax, title=f"Targeting by predicted CATE beats random (Qini = {qini['qini_coefficient']:.1f})",
             subtitle="Cumulative incremental conversions vs. share of test population targeted",
             xlabel="Share of population targeted (%)", ylabel="Cumulative incremental conversions")
    ax.legend(fontsize=9.5, loc="upper left")
    savefig(fig, FIG_DIR / "uplift_qini_curve.png", footnote=source_note)

    # --- What's actually driving the heterogeneity: predicted CATE by tenure ---
    test_df["tenure_bucket"] = pd.qcut(test_df["tenure_days"], 6, duplicates="drop")
    by_tenure = test_df.groupby("tenure_bucket", observed=True)["cate_pred"].mean()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in by_tenure.index]
    ax.plot(range(len(by_tenure)), by_tenure.values * 100, marker="o", markersize=6, color=MUTED_RED, linewidth=1.8)
    ax.set_xticks(range(len(by_tenure)))
    ax.set_xticklabels(labels, fontsize=9.5)
    style_ax(ax, title="The model recovers tenure as the driver of heterogeneity",
             subtitle="Mean predicted CATE by platform-tenure bucket (days), test set",
             xlabel="Tenure (days)", ylabel="Predicted CATE (pp)")
    savefig(fig, FIG_DIR / "uplift_by_tenure.png", footnote=source_note)

    print(f"Train: {len(train_df):,} | Test (held out, scored only): {len(test_df):,}")
    print("Predicted-CATE quintile calibration (quintile 1 = highest predicted uplift):")
    print(bucket_df[["bucket", "n", "predicted_cate_mean", "realized_lift"]].round(4).to_string(index=False))
    print(f"Qini coefficient: {qini['qini_coefficient']:.2f}")
    print("Mean predicted CATE by tenure bucket:")
    print((by_tenure * 100).round(2).to_string())
    return {"bucket_df": bucket_df, "qini_coefficient": qini["qini_coefficient"], "by_tenure": by_tenure}


def main():
    df = pd.read_csv(DATA_DIR / "experiment_users.csv")
    source_note = f"Source: synthetic BNPL experiment data · n = {len(df):,} users, 60/40 train/test split"
    uplift_modeling(df, source_note)
    print("Wrote reports/figures/uplift_calibration.png, uplift_qini_curve.png, uplift_by_tenure.png")


if __name__ == "__main__":
    main()
