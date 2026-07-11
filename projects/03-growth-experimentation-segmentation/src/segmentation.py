"""
RFM (recency, frequency, monetary) customer segmentation via KMeans.

Log-transforms and standardizes the three RFM dimensions (their raw
scales are wildly different and right-skewed, which would otherwise let
monetary value dominate the distance metric), picks k via silhouette
score, then labels each resulting cluster using its RFM profile rather
than a fixed a-priori taxonomy.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_AMBER, MUTED_RED, GREY, PALETTE

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

K_RANGE = range(2, 8)


def choose_k(X, source_note):
    inertias, sil_scores = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=7, n_init=10).fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(list(K_RANGE), inertias, color=SLATE, marker="o", markersize=5, linewidth=1.6)
    style_ax(axes[0], title="Elbow method", xlabel="Number of clusters (k)", ylabel="Inertia (within-cluster SS)")

    best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
    colors = [MUTED_RED if k == best_k else SLATE for k in K_RANGE]
    axes[1].bar([str(k) for k in K_RANGE], sil_scores, color=colors, zorder=3, width=0.55)
    style_ax(axes[1], title="Silhouette score", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    plt.tight_layout()
    savefig(fig, FIG_DIR / "segmentation_k_selection.png", footnote=source_note)

    print(f"Silhouette scores by k: {dict(zip(K_RANGE, [round(s, 3) for s in sil_scores]))}")
    print(f"Selected k={best_k}")
    return best_k


NAME_SETS = {
    2: ["Champions", "Dormant"],
    3: ["Champions", "Loyal", "Dormant"],
    4: ["Champions", "Loyal", "At risk", "Dormant"],
    5: ["Champions", "Loyal", "Promising", "At risk", "Dormant"],
    6: ["Champions", "Loyal", "Promising", "At risk", "Lapsed", "Dormant"],
}


def label_segments(profile):
    """Rank-based labeling from the cluster-mean RFM profile (best to
    worst combined RFM rank), mapped onto a name set sized to however
    many clusters were actually found, so labels always span the full
    best-to-worst spectrum instead of bunching at one end."""
    r_rank = profile["recency_days"].rank()       # low recency (rank 1) = most recent
    f_rank = profile["frequency"].rank(ascending=False)
    m_rank = profile["monetary_usd"].rank(ascending=False)
    score = r_rank + f_rank + m_rank
    order = score.sort_values().index

    n = len(order)
    names = NAME_SETS.get(n, [f"Segment {i+1}" for i in range(n)])
    return {seg: names[i] for i, seg in enumerate(order)}


def main():
    df = pd.read_csv(DATA_DIR / "rfm_customers.csv")
    n_customers = len(df)
    source_note = f"Source: synthetic BNPL customer data · n = {n_customers:,} customers"

    rfm = df[["customer_id", "recency_days", "frequency", "monetary_usd"]].copy()
    log_features = np.column_stack([
        np.log1p(rfm["recency_days"]), np.log1p(rfm["frequency"]), np.log1p(rfm["monetary_usd"]),
    ])
    X = StandardScaler().fit_transform(log_features)

    best_k = choose_k(X, source_note)
    km = KMeans(n_clusters=best_k, random_state=7, n_init=10).fit(X)
    rfm["cluster"] = km.labels_

    profile = rfm.groupby("cluster")[["recency_days", "frequency", "monetary_usd"]].mean()
    label_map = label_segments(profile)
    rfm["segment"] = rfm["cluster"].map(label_map)
    profile.index = profile.index.map(label_map)
    profile = profile.reindex(NAME_SETS[best_k])

    rfm.to_csv(BASE / "reports" / "rfm_segments.csv", index=False)

    # --- Segment profile: raw mean values, one panel per RFM dimension.
    # Min-max normalizing three metrics onto one 0-1 axis forces
    # whichever segment is worst on every dimension down to a flat zero,
    # which reads as missing data rather than "worst". Small multiples
    # on each metric's own scale avoid that and are just as easy to read.
    panels = [("recency_days", "Recency", "days since last order (lower = more recent)"),
              ("frequency", "Frequency", "orders in the observation window"),
              ("monetary_usd", "Monetary", "avg. spend (USD)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (col, label, sublabel) in zip(axes, panels):
        ax.bar(profile.index, profile[col], color=PALETTE[:len(profile)], width=0.6, zorder=3)
        style_ax(ax, title=label, subtitle=sublabel, grid_axis="y")
        ax.tick_params(axis="x", labelsize=9.5)
    fig.suptitle("Each segment has a distinct recency, frequency, and value profile",
                 x=0.01, y=1.06, ha="left", fontsize=14.5, fontfamily="Lora", color="#2B2B2B")
    plt.tight_layout()
    savefig(fig, FIG_DIR / "segment_rfm_profile.png", footnote=source_note)

    # --- Segment size vs. revenue contribution ---
    seg_summary = rfm.groupby("segment").agg(customers=("customer_id", "count"), revenue=("monetary_usd", "sum"))
    seg_summary = seg_summary.reindex(profile.index)
    seg_summary["pct_customers"] = seg_summary["customers"] / seg_summary["customers"].sum() * 100
    seg_summary["pct_revenue"] = seg_summary["revenue"] / seg_summary["revenue"].sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(seg_summary))
    width = 0.35
    ax.bar(x - width/2, seg_summary["pct_customers"], width, color=GREY, label="% of customers", zorder=3)
    ax.bar(x + width/2, seg_summary["pct_revenue"], width, color=SLATE, label="% of revenue", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_summary.index)
    style_ax(ax, title="Champions and Loyal customers punch well above their headcount",
             subtitle="Share of customer base vs. share of total revenue, by segment",
             ylabel="Share (%)")
    ax.legend(loc="upper right", fontsize=9.5)
    savefig(fig, FIG_DIR / "segment_revenue_share.png", footnote=source_note)

    print(profile.round(1))
    print(seg_summary.round(1))
    print("Wrote reports/rfm_segments.csv and 3 figures.")


if __name__ == "__main__":
    main()
