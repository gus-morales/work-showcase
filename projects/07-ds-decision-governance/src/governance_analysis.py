"""
How impact level relates to approval speed and to what happens after a
decision ships. Two descriptive charts (approval lag, rollback rate,
both by impact level), plus an optional logistic regression that checks
whether those same relationships hold once impact level, artifact type,
and approval speed are all considered together instead of one at a time.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

IMPACT_ORDER = ["low", "medium", "high"]


def approval_lag_chart(df, source_note):
    lag = df.groupby("impact_level", observed=True)["approval_lag_days"].mean().reindex(IMPACT_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(lag.index, lag.values, color=SLATE, width=0.5, zorder=3)
    for i, v in enumerate(lag.values):
        ax.text(i, v + lag.max() * 0.02, f"{v:.1f}d", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Higher-impact decisions take longer to approve",
             subtitle="Mean approval lag (proposed to approved) by impact level",
             xlabel="Impact level", ylabel="Approval lag (days)")
    savefig(fig, FIG_DIR / "approval_lag_by_impact.png", footnote=source_note)
    return lag


def rollback_rate_chart(df, source_note):
    resolved = df[df["status"].isin(["closed", "reverted"])]
    rate = resolved.groupby("impact_level", observed=True)["outcome"].apply(
        lambda s: (s == "rollback").mean()
    ).reindex(IMPACT_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(rate.index, rate.values * 100, color=MUTED_RED, width=0.5, zorder=3)
    for i, v in enumerate(rate.values):
        ax.text(i, v * 100 + 0.4, f"{v:.1%}", ha="center", fontsize=10.5, color=INK)
    style_ax(ax, title="Rollback rate falls as impact level rises",
             subtitle="Share of resolved decisions later rolled back, by impact level",
             xlabel="Impact level", ylabel="Rollback rate (%)")
    savefig(fig, FIG_DIR / "rollback_rate_by_impact.png", footnote=source_note)
    return rate


def rollback_logistic_regression(df, source_note):
    """Optional interpretability check: does impact level still predict
    rollback once artifact type and approval lag are in the same model,
    instead of looked at one at a time? Fit on resolved decisions only
    (closed or reverted); abandoned decisions never reached an outcome."""
    resolved = df[df["status"].isin(["closed", "reverted"])].copy()
    resolved["rollback"] = (resolved["outcome"] == "rollback").astype(int)

    X = pd.get_dummies(resolved[["impact_level", "artifact_type"]], drop_first=True)
    X["approval_lag_days"] = resolved["approval_lag_days"].values
    feature_names = X.columns.tolist()
    y = resolved["rollback"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_s, y_train)
    pr_auc = average_precision_score(y_test, model.predict_proba(X_test_s)[:, 1])

    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [MUTED_RED if c > 0 else SLATE for c in coefs.values]
    ax.barh(coefs.index.str.replace("_", " "), coefs.values, color=colors, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.8)
    style_ax(ax, title="Approval lag carries most of the signal, since impact level set it in the first place",
             subtitle="Standardized logistic regression coefficients, rollback vs. keep/iterate",
             xlabel="Coefficient (positive = higher rollback odds)", grid_axis="x")
    savefig(fig, FIG_DIR / "rollback_logistic_coefficients.png", footnote=source_note)

    return {"pr_auc": pr_auc, "base_rate": float(y_test.mean()), "coefficients": coefs.to_dict()}


def main():
    df = pd.read_csv(BASE / "data" / "decision_log.csv")
    df["impact_level"] = pd.Categorical(df["impact_level"], categories=IMPACT_ORDER, ordered=True)
    source_note = f"Source: synthetic decision log (src/generate_data.py) · n = {len(df):,} decisions"

    lag = approval_lag_chart(df, source_note)
    rate = rollback_rate_chart(df, source_note)
    lr_result = rollback_logistic_regression(df, source_note)

    print(f"Mean approval lag by impact level:\n{lag.round(2)}")
    print(f"Rollback rate by impact level:\n{rate.round(3)}")
    print(f"Logistic regression PR-AUC: {lr_result['pr_auc']:.3f} (base rate {lr_result['base_rate']:.3f})")
    print(f"Coefficients:\n{pd.Series(lr_result['coefficients']).round(3)}")
    print("Wrote reports/figures/approval_lag_by_impact.png, rollback_rate_by_impact.png, "
          "rollback_logistic_coefficients.png")

    metrics = {
        "n_decisions": int(len(df)),
        "approval_lag_days_mean_by_impact": lag.round(2).to_dict(),
        "rollback_rate_by_impact": rate.round(4).to_dict(),
        "logistic_regression": {
            "pr_auc": round(lr_result["pr_auc"], 4),
            "base_rate": round(lr_result["base_rate"], 4),
            "coefficients": {k: round(v, 4) for k, v in lr_result["coefficients"].items()},
        },
    }
    (BASE / "reports" / "governance_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
