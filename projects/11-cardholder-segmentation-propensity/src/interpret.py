"""SHAP-based interpretability for the trained propensity GBM: checks
that the model actually leans on baseline-value proxies and the recency
sweet spot, the same signals the data was generated from, and that the
decoy decline-rate feature doesn't rank highly."""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from propensity_model import build_features
from style import set_style, add_footnote, INK

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
set_style()


def main():
    customers = pd.read_csv(BASE / "data" / "customers.csv")
    segments = pd.read_csv(BASE / "reports" / "customer_segments.csv")
    df = customers.merge(segments[["customer_id", "segment"]], on="customer_id", how="left")

    bundle = joblib.load(BASE / "reports" / "model.pkl")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    offered = df[df["past_offer_sent"] == 1]
    X_offered = build_features(offered).reindex(columns=feature_columns, fill_value=0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_offered)

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_offered, show=False, max_display=15, plot_size=None)
    ax = plt.gca()
    ax.set_title("SHAP feature importance", loc="left", fontsize=14.5,
                 fontweight="normal", fontfamily="Lora", color=INK, pad=14)
    ax.set_xlabel("SHAP value (impact on predicted response propensity)", fontsize=11, labelpad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    add_footnote(fig, f"Source: TreeExplainer on the trained GBM · offered customers · n = {len(X_offered):,}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=170, bbox_inches="tight")
    plt.close()

    mean_abs = pd.Series(
        abs(shap_values.values).mean(axis=0), index=feature_columns,
    ).sort_values(ascending=False)

    top10 = mean_abs.head(10)
    decline_rank = int(mean_abs.rank(ascending=False)["decline_rate"])
    lines = ["# SHAP feature importance (mean |SHAP value|, offered customers)\n"]
    lines.append(top10.round(4).to_markdown())
    lines.append(f"\n\n`decline_rate` (the decoy feature) ranks {decline_rank} of {len(mean_abs)}.\n")
    (BASE / "reports" / "shap_top_features.md").write_text("\n".join(lines))

    print(top10)
    print(f"decline_rate ranks {decline_rank} of {len(mean_abs)}")
    print("Saved reports/figures/shap_summary.png and reports/shap_top_features.md")


if __name__ == "__main__":
    main()
