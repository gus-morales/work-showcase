"""SHAP-based interpretability for the trained fraud GBM."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from features import temporal_split, RAW_FEATURE_COLS
from style import set_style, add_footnote, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
set_style()


def main():
    df = pd.read_csv(BASE / "data" / "transactions.csv", parse_dates=["timestamp"])
    _, _, test_df = temporal_split(df)

    bundle = joblib.load(BASE / "reports" / "model.pkl")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    # Transform only, with the pipeline fit on the training split by
    # train.py: SHAP explains the model exactly as it was trained, not a
    # version re-fit on the test window.
    X_test = bundle["feature_pipeline"].transform(test_df[RAW_FEATURE_COLS])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15, plot_size=None)
    ax = plt.gca()
    ax.set_title("SHAP feature importance", loc="left", fontsize=14.5,
                 fontweight="normal", fontfamily="Lora", color=INK, pad=14)
    ax.set_xlabel("SHAP value (impact on predicted fraud risk)", fontsize=11, labelpad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    add_footnote(fig, f"Source: TreeExplainer on the trained GBM · held-out test set · n = {len(test_df):,} transactions")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=170, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(
        abs(shap_values.values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)

    top10 = mean_abs.head(10)
    lines = ["# SHAP feature importance (mean |SHAP value|, held-out test set)\n"]
    lines.append(top10.round(4).to_markdown())
    (BASE / "reports" / "shap_top_features.md").write_text("\n".join(lines))

    print(top10)
    print("Saved reports/figures/shap_summary.png and reports/shap_top_features.md")


if __name__ == "__main__":
    main()
