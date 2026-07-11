"""SHAP-based interpretability for the trained GBM model."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from features import build_design_matrix, engineer_features
from style import set_style

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
set_style()


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")
    df = engineer_features(df)
    test_df = df[df.origination_month > 21]
    X_all, feature_names = build_design_matrix(df)
    X_test = X_all.loc[test_df.index]

    bundle = joblib.load(BASE / "reports" / "model.pkl")
    model = bundle["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15, plot_size=None)
    ax = plt.gca()
    ax.set_title("SHAP feature importance", loc="left", fontsize=14.5,
                 fontweight="normal", fontfamily="Lora", color="#2B2B2B", pad=14)
    ax.set_xlabel("SHAP value (impact on predicted risk)", fontsize=11, labelpad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=170, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(
        abs(shap_values.values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)

    top10 = mean_abs.head(10)
    lines = ["# SHAP feature importance (mean |SHAP value|, held-out months 22-24)\n"]
    lines.append(top10.round(4).to_markdown())
    (BASE / "reports" / "shap_top_features.md").write_text("\n".join(lines))

    print(top10)
    print("Saved reports/figures/shap_summary.png and reports/shap_top_features.md")


if __name__ == "__main__":
    main()
