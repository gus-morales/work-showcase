"""Exploratory analysis: saves charts to reports/figures and a numeric
summary to reports/eda_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from style import set_style, style_ax, savefig, PALETTE, NAVY, TEAL, CORAL

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")

    # 1. Delinquency rate by month (shows the late-window shock)
    by_month = df.groupby("origination_month")["delinquent_30dpd"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(by_month.index, by_month.values * 100, marker="o", color=NAVY, linewidth=2.2, markersize=6)
    ax.axvspan(21.5, 24, color=CORAL, alpha=0.10)
    ax.text(22.7, by_month.values.max() * 100 * 1.05, "macro shock\nwindow", fontsize=10,
            color=CORAL, ha="center", fontweight="bold")
    style_ax(ax, title="Delinquency climbs sharply in the last 3 months",
             subtitle="30+ DPD rate by loan origination month",
             xlabel="Origination month", ylabel="30+ DPD rate (%)")
    savefig(fig, FIG_DIR / "delinquency_by_month.png")

    # 2. Delinquency by employment type
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df.groupby("employment_type")["delinquent_30dpd"].mean().sort_values(ascending=False).index
    rates = df.groupby("employment_type")["delinquent_30dpd"].mean().reindex(order) * 100
    labels = [o.replace("_", " ").title() for o in order]
    bars = ax.bar(labels, rates.values, color=PALETTE[:len(order)], width=0.6, zorder=3)
    for b, v in zip(bars, rates.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10.5, color="#333")
    style_ax(ax, title="Informal and gig-economy borrowers default more often",
             subtitle="30+ DPD rate by employment type",
             ylabel="30+ DPD rate (%)")
    savefig(fig, FIG_DIR / "delinquency_by_employment.png")

    # 3. Bureau score distribution by outcome
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="credit_bureau_score", hue="delinquent_30dpd", fill=True,
                common_norm=False, palette=[TEAL, CORAL], alpha=0.45, linewidth=1.5, ax=ax, legend=False)
    ax.plot([], [], color=TEAL, linewidth=6, alpha=0.45, label="Current")
    ax.plot([], [], color=CORAL, linewidth=6, alpha=0.45, label="30+ DPD")
    ax.legend(loc="upper left")
    style_ax(ax, title="Delinquent borrowers skew toward lower bureau scores",
             subtitle="Credit bureau score distribution by outcome",
             xlabel="Credit bureau score", ylabel="Density")
    savefig(fig, FIG_DIR / "bureau_score_by_outcome.png")

    # 4. Loan amount vs income ratio vs outcome
    df["loan_to_income_ratio"] = df["loan_amount_mxn"] / df["monthly_income_mxn"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="delinquent_30dpd", y="loan_to_income_ratio", ax=ax,
                palette=[TEAL, CORAL], showfliers=False, width=0.5, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Current", "30+ DPD"])
    style_ax(ax, title="Bigger loans relative to income default more",
             subtitle="Loan amount ÷ monthly income, by outcome",
             xlabel="", ylabel="Loan amount / monthly income")
    savefig(fig, FIG_DIR / "loan_to_income_by_outcome.png")

    # Summary markdown
    lines = ["# EDA summary\n"]
    lines.append(f"- Loans: {len(df):,}, Customers: {df['customer_id'].nunique():,}\n")
    lines.append(f"- Overall 30+ DPD rate: {df['delinquent_30dpd'].mean():.2%}\n")
    lines.append(f"- Months 1-21 rate: {df.loc[df.origination_month < 22, 'delinquent_30dpd'].mean():.2%}, "
                 f"months 22-24 (shock window) rate: {df.loc[df.origination_month >= 22, 'delinquent_30dpd'].mean():.2%}\n")
    lines.append("\n## Delinquency rate by employment type\n")
    lines.append(df.groupby("employment_type")["delinquent_30dpd"].mean().sort_values(ascending=False)
                 .mul(100).round(2).astype(str).add(" %").to_markdown())
    lines.append("\n\n## Delinquency rate by merchant category\n")
    lines.append(df.groupby("merchant_category")["delinquent_30dpd"].mean().sort_values(ascending=False)
                 .mul(100).round(2).astype(str).add(" %").to_markdown())
    (BASE / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("EDA complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
