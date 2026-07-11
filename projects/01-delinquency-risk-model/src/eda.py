"""Exploratory analysis: saves charts to reports/figures and a numeric
summary to reports/eda_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from style import set_style, style_ax, savefig, PALETTE, SLATE, MUTED_TEAL, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")
    SOURCE = f"Source: synthetic BNPL loan data (src/generate_data.py) · n = {len(df):,} loans"

    # 1. Delinquency rate by month (shows the late-window shock)
    by_month = df.groupby("origination_month")["delinquent_30dpd"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(by_month.index, by_month.values * 100, color=SLATE, linewidth=1.6)
    ax.scatter(by_month.index, by_month.values * 100, color=SLATE, s=18, zorder=3)
    ax.axvspan(21.5, 24, color=GREY, alpha=0.08)
    ax.text(22.7, 1.2, "shock window", fontsize=9, color=GREY, ha="center", style="italic")
    style_ax(ax, title="Delinquency rate by origination month",
             xlabel="Origination month", ylabel="30+ DPD rate (%)")
    ax.set_ylim(0, by_month.values.max() * 100 * 1.15)
    savefig(fig, FIG_DIR / "delinquency_by_month.png", footnote=SOURCE)

    # 2. Delinquency by employment type
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df.groupby("employment_type")["delinquent_30dpd"].mean().sort_values(ascending=False).index
    rates = df.groupby("employment_type")["delinquent_30dpd"].mean().reindex(order) * 100
    labels = [o.replace("_", " ").title() for o in order]
    ax.bar(labels, rates.values, color=SLATE, width=0.55, zorder=3)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10, color="#4a4a4a")
    style_ax(ax, title="Delinquency rate by employment type",
             ylabel="30+ DPD rate (%)")
    savefig(fig, FIG_DIR / "delinquency_by_employment.png", footnote=SOURCE)

    # 3. Bureau score distribution by outcome
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="credit_bureau_score", hue="delinquent_30dpd", fill=True,
                common_norm=False, palette=[MUTED_TEAL, MUTED_RED], alpha=0.35, linewidth=1.3, ax=ax, legend=False)
    ax.plot([], [], color=MUTED_TEAL, linewidth=6, alpha=0.35, label="Current")
    ax.plot([], [], color=MUTED_RED, linewidth=6, alpha=0.35, label="30+ DPD")
    ax.legend(loc="upper left")
    style_ax(ax, title="Credit bureau score distribution by outcome",
             xlabel="Credit bureau score", ylabel="Density")
    savefig(fig, FIG_DIR / "bureau_score_by_outcome.png", footnote=SOURCE)

    # 4. Loan amount vs income ratio vs outcome
    df["loan_to_income_ratio"] = df["loan_amount_mxn"] / df["monthly_income_mxn"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="delinquent_30dpd", y="loan_to_income_ratio", hue="delinquent_30dpd",
                ax=ax, palette=[MUTED_TEAL, MUTED_RED], showfliers=False, width=0.45, zorder=3, legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Current", "30+ DPD"])
    style_ax(ax, title="Loan-to-income ratio by outcome",
             xlabel="", ylabel="Loan amount / monthly income")
    savefig(fig, FIG_DIR / "loan_to_income_by_outcome.png", footnote=SOURCE)

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
