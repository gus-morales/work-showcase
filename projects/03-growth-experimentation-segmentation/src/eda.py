"""Exploratory analysis across all four datasets in this project, before
any test is read, any rollout is evaluated, or any customer is clustered:
saves charts to reports/figures and a numeric summary to
reports/eda_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_RED, INK

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()

TOPIC_LABELS = {
    "late_fee_dispute": "Late fee dispute", "app_bug": "App bug",
    "kyc_verification": "KYC verification", "refund_request": "Refund request",
    "general_inquiry": "General inquiry",
}


def main():
    exp = pd.read_csv(DATA_DIR / "experiment_users.csv")
    rollout = pd.read_csv(DATA_DIR / "regional_rollout.csv")
    rfm = pd.read_csv(DATA_DIR / "rfm_customers.csv")
    tickets = pd.read_csv(DATA_DIR / "support_tickets.csv")

    # 1. A/B test: randomization check on a pre-treatment covariate. If the
    # split worked, the two arms should look statistically identical on
    # anything measured before assignment.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=exp, x="revenue_pre_30d_usd", hue="arm", fill=True, common_norm=False,
                palette=[MUTED_TEAL, MUTED_RED], alpha=0.35, linewidth=1.3, ax=ax, legend=False)
    ax.plot([], [], color=MUTED_TEAL, linewidth=6, alpha=0.35, label="Control")
    ax.plot([], [], color=MUTED_RED, linewidth=6, alpha=0.35, label="Treatment")
    ax.legend(loc="upper right")
    style_ax(ax, title="Control and treatment look identical before the test even starts",
             subtitle="Pre-period revenue (the CUPED covariate), by assigned arm",
             xlabel="Revenue in the 30 days before assignment (USD)", ylabel="Density")
    savefig(fig, FIG_DIR / "ab_balance_check.png",
            footnote=f"Source: synthetic BNPL experiment data (src/generate_data.py) · n = {len(exp):,} users")

    # 2. Regional rollout: per-region pre-rollout baseline rate. Regions
    # vary meaningfully on their own before any rollout happens, which is
    # exactly why the DiD regression below controls for region fixed
    # effects instead of comparing raw treated-vs-control levels.
    pre = rollout[rollout["post_rollout"] == False]  # noqa: E712
    region_baseline = pre.groupby(["region_id", "group"], as_index=False)["on_time_rate"].mean()
    region_baseline = region_baseline.sort_values("on_time_rate")
    colors = [MUTED_RED if g == "treated" else SLATE for g in region_baseline["group"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(range(len(region_baseline)), region_baseline["on_time_rate"] * 100, color=colors, width=0.7, zorder=3)
    ax.plot([], [], color=SLATE, linewidth=6, label="Control")
    ax.plot([], [], color=MUTED_RED, linewidth=6, label="Treated")
    ax.legend(loc="upper left")
    ax.set_xticks([])
    style_ax(ax, title="Regions vary a lot on their own, before any rollout",
             subtitle="Average on-time repayment rate before rollout day, by region",
             xlabel="Region (sorted by baseline rate)", ylabel="Pre-rollout on-time rate (%)")
    savefig(fig, FIG_DIR / "regional_baseline_by_region.png",
            footnote=f"Source: synthetic BNPL regional panel (src/generate_data.py) · n = {rollout['region_id'].nunique()} regions")

    # 3. RFM: raw recency/frequency scatter, before any clustering. The
    # four latent archetypes should already be visible as separated
    # clouds, which is what gives KMeans real structure to recover rather
    # than an arbitrary cut through a single blob.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(rfm["recency_days"], rfm["frequency"], s=8, alpha=0.25, color=SLATE, linewidths=0)
    style_ax(ax, title="Customers already form distinct clusters, before any labels exist",
             subtitle="Recency vs. frequency, one dot per customer",
             xlabel="Recency (days since last order)", ylabel="Frequency (orders in window)")
    savefig(fig, FIG_DIR / "rfm_recency_frequency_scatter.png",
            footnote=f"Source: synthetic BNPL customer data (src/generate_data.py) · n = {len(rfm):,} customers")

    # 4. Support tickets: true topic distribution, before topic modeling
    # tries to recover it unsupervised. Also sets up expectations for the
    # "how close does NMF get" comparison later.
    dist = tickets["true_topic"].value_counts(normalize=True).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [TOPIC_LABELS[t] for t in dist.index]
    ax.bar(labels, dist.values * 100, color=SLATE, width=0.55, zorder=3)
    for i, v in enumerate(dist.values):
        ax.text(i, v * 100 + 0.6, f"{v:.1%}", ha="center", fontsize=10, color=INK)
    style_ax(ax, title="Late fee disputes are the single largest ticket category",
             subtitle="Share of tickets by underlying topic (ground truth, for validation only)",
             ylabel="Share of tickets (%)")
    savefig(fig, FIG_DIR / "ticket_topic_distribution.png",
            footnote=f"Source: synthetic BNPL support tickets (src/generate_data.py) · n = {len(tickets):,} tickets")

    # Summary markdown
    bal = exp.groupby("arm")[["tenure_days", "sessions_pre_30d", "revenue_pre_30d_usd"]].mean().round(2)
    lines = ["# EDA summary\n"]
    lines.append("## A/B test covariate balance (control vs. treatment)\n")
    lines.append(bal.to_markdown())
    lines.append(f"\n\n- Regional panel: {rollout['region_id'].nunique()} regions, "
                 f"pre-rollout on-time rate {pre['on_time_rate'].mean():.1%} overall, "
                 f"ranging {region_baseline['on_time_rate'].min():.1%} to {region_baseline['on_time_rate'].max():.1%} "
                 f"across individual regions\n")
    lines.append(f"- RFM customers: {len(rfm):,}, recency {rfm['recency_days'].median():.0f} days "
                 f"(median), frequency {rfm['frequency'].median():.0f} orders (median), "
                 f"monetary ${rfm['monetary_usd'].median():,.0f} (median)\n")
    lines.append(f"- Support tickets: {len(tickets):,}, "
                 f"{tickets['ticket_text'].str.split().str.len().mean():.1f} words on average\n")
    (BASE / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("EDA complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
