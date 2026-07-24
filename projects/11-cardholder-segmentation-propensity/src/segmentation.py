"""
Cardholder segmentation via a Gaussian Mixture Model on trailing-90-day
behavioral features (recency, frequency, monetary, category diversity,
checkout decline rate). Log-transforms and standardizes first: raw
scales are wildly different and right-skewed, which would otherwise
let monetary value dominate distance/likelihood purely by having the
largest raw scale.

Component count is chosen by BIC, a likelihood-based model-selection
criterion, since GMM gives a proper log-likelihood to penalize, unlike
a hard k-means partition.

Deliberately excludes tenure and lifetime order count: this segments
customers on CURRENT behavior only, the kind of feature set a growth
team would build a segmentation off first. Whether that's enough to
find who's worth a win-back offer is exactly what the propensity model
in propensity_model.py checks next.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY, PALETTE, INK

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

K_RANGE = range(2, 8)
MAX_INTERPRETABLE_K = 6  # see choose_k(): BIC never elbows within K_RANGE
SEGMENT_FEATURES = ["recency_days", "frequency_90d", "monetary_90d", "category_diversity", "decline_rate"]


def choose_k(X, source_note):
    bics, aics = [], []
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, random_state=7, n_init=5).fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    # BIC keeps falling all the way to the edge of K_RANGE with no clean
    # elbow, typical once there's enough data that BIC's complexity
    # penalty barely bites: more components almost always buy a little
    # more likelihood. Cap the candidate range at a size a growth team
    # could actually act on, and take the minimum within that cap
    # instead of the unconstrained minimum.
    candidates = [k for k in K_RANGE if k <= MAX_INTERPRETABLE_K]
    candidate_bics = [bics[list(K_RANGE).index(k)] for k in candidates]
    best_k = candidates[int(np.argmin(candidate_bics))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_bic = [MUTED_RED if k == best_k else SLATE for k in K_RANGE]
    axes[0].bar([str(k) for k in K_RANGE], bics, color=colors_bic, zorder=3, width=0.55)
    style_ax(axes[0], title="BIC (lower is better)",
             subtitle=f"No clean elbow past k={MAX_INTERPRETABLE_K}; candidates capped there",
             xlabel="Number of components (k)", ylabel="BIC")

    axes[1].plot(list(K_RANGE), aics, color=SLATE, marker="o", markersize=5, linewidth=1.6)
    style_ax(axes[1], title="AIC, shown for comparison", xlabel="Number of components (k)", ylabel="AIC")
    plt.tight_layout()
    savefig(fig, FIG_DIR / "segmentation_k_selection.png", footnote=source_note)

    print(f"BIC by k: {dict(zip(K_RANGE, [round(b, 1) for b in bics]))}")
    print(f"Selected k={best_k} (min BIC among k<={MAX_INTERPRETABLE_K})")
    return best_k


NAME_SETS = {
    2: ["High-Value Active", "Lapsed/Dormant"],
    3: ["High-Value Active", "Steady Regular", "Lapsed/Dormant"],
    4: ["High-Value Active", "Steady Regular", "Lapsed", "Dormant"],
    5: ["High-Value Active", "Steady Regular", "Occasional", "Lapsed", "Dormant"],
    6: ["High-Value Active", "Steady Regular", "Occasional", "Light Spender", "Lapsed", "Dormant"],
}


def label_segments(profile):
    """Rank-based labeling from the cluster-mean behavioral profile
    (best to worst combined recency/frequency/monetary rank), mapped
    onto a name set sized to however many components BIC actually
    picked, so labels always span the full best-to-worst spectrum."""
    r_rank = profile["recency_days"].rank()  # low recency (rank 1) = most recent = best
    f_rank = profile["frequency_90d"].rank(ascending=False)
    m_rank = profile["monetary_90d"].rank(ascending=False)
    score = r_rank + f_rank + m_rank
    order = score.sort_values().index

    n = len(order)
    names = NAME_SETS.get(n, [f"Segment {i + 1}" for i in range(n)])
    return {seg: names[i] for i, seg in enumerate(order)}


def main():
    df = pd.read_csv(DATA_DIR / "customers.csv")
    n_customers = len(df)
    source_note = f"Source: synthetic bank customer data · n = {n_customers:,} customers"

    log_features = np.column_stack([
        np.log1p(df["recency_days"]), np.log1p(df["frequency_90d"]), np.log1p(df["monetary_90d"]),
        df["category_diversity"], df["decline_rate"],
    ])
    X = StandardScaler().fit_transform(log_features)

    best_k = choose_k(X, source_note)
    gmm = GaussianMixture(n_components=best_k, random_state=7, n_init=5).fit(X)
    df["cluster"] = gmm.predict(X)

    profile = df.groupby("cluster")[SEGMENT_FEATURES].mean()
    label_map = label_segments(profile)
    df["segment"] = df["cluster"].map(label_map)
    profile.index = profile.index.map(label_map)
    profile = profile.reindex(NAME_SETS[best_k])

    df[["customer_id", "cluster", "segment"]].to_csv(BASE / "reports" / "customer_segments.csv", index=False)

    # --- Segment profile: raw mean values, one panel per behavioral dimension ---
    panels = [("recency_days", "Recency", "days since last order (lower = more recent)"),
              ("frequency_90d", "Frequency", "orders in the trailing 90-day window"),
              ("monetary_90d", "Monetary", "avg. 90-day spend (USD)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (col, label, sublabel) in zip(axes, panels):
        ax.bar(profile.index, profile[col], color=PALETTE[:len(profile)], width=0.6, zorder=3)
        style_ax(ax, title=label, subtitle=sublabel, grid_axis="y")
        ax.tick_params(axis="x", labelsize=9, rotation=15)
    fig.suptitle("Each segment has a distinct recency, frequency, and value profile",
                 x=0.01, y=1.06, ha="left", fontsize=14.5, fontfamily="Lora", color=INK)
    plt.tight_layout()
    savefig(fig, FIG_DIR / "segment_behavior_profile.png", footnote=source_note)

    # --- Segment size vs. spend contribution ---
    seg_summary = df.groupby("segment").agg(customers=("customer_id", "count"), spend=("monetary_90d", "sum"))
    seg_summary = seg_summary.reindex(profile.index)
    seg_summary["pct_customers"] = seg_summary["customers"] / seg_summary["customers"].sum() * 100
    seg_summary["pct_spend"] = seg_summary["spend"] / seg_summary["spend"].sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(seg_summary))
    width = 0.35
    ax.bar(x - width / 2, seg_summary["pct_customers"], width, color=GREY, label="% of customers", zorder=3)
    ax.bar(x + width / 2, seg_summary["pct_spend"], width, color=SLATE, label="% of 90-day spend", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_summary.index, rotation=15, ha="right", fontsize=9)
    style_ax(ax, title="High-Value Active customers punch well above their headcount",
             subtitle="Share of customer base vs. share of total 90-day spend, by segment",
             ylabel="Share (%)")
    ax.legend(loc="upper right", fontsize=9.5)
    savefig(fig, FIG_DIR / "segment_spend_share.png", footnote=source_note)

    print(profile.round(2))
    print(seg_summary.round(1))
    print(f"Wrote reports/customer_segments.csv and figures for k={best_k} segments.")


if __name__ == "__main__":
    main()
