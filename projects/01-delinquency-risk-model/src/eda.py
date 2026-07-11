"""Exploratory analysis: saves charts to reports/figures and a numeric
summary to reports/eda_summary.md."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = ["#1f3b57", "#2f7d9e", "#57b8b0", "#f2a541", "#d9544d"]


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")

    # 1. Delinquency rate by month (shows the late-window shock)
    by_month = df.groupby("origination_month")["delinquent_30dpd"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(by_month.index, by_month.values * 100, marker="o", color=PALETTE[0])
    ax.axvspan(22, 24, color=PALETTE[4], alpha=0.15, label="macro shock window")
    ax.set_xlabel("Origination month")
    ax.set_ylabel("30+ DPD rate (%)")
    ax.set_title("Delinquency rate by origination month")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delinquency_by_month.png", dpi=150)
    plt.close(fig)

    # 2. Delinquency by employment type
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df.groupby("employment_type")["delinquent_30dpd"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="employment_type", y="delinquent_30dpd", order=order, ax=ax,
                palette=PALETTE, errorbar=None)
    ax.set_ylabel("30+ DPD rate")
    ax.set_xlabel("Employment type")
    ax.set_title("Delinquency rate by employment type")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delinquency_by_employment.png", dpi=150)
    plt.close(fig)

    # 3. Bureau score distribution by outcome
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="credit_bureau_score", hue="delinquent_30dpd", fill=True,
                common_norm=False, palette=[PALETTE[1], PALETTE[4]], ax=ax)
    ax.set_title("Credit bureau score distribution by outcome")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bureau_score_by_outcome.png", dpi=150)
    plt.close(fig)

    # 4. Loan amount vs income ratio vs outcome
    df["loan_to_income_ratio"] = df["loan_amount_mxn"] / df["monthly_income_mxn"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="delinquent_30dpd", y="loan_to_income_ratio", ax=ax,
                palette=[PALETTE[1], PALETTE[4]], showfliers=False)
    ax.set_xticklabels(["Current", "30+ DPD"])
    ax.set_xlabel("")
    ax.set_ylabel("Loan amount / monthly income")
    ax.set_title("Loan-to-income ratio by outcome")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loan_to_income_by_outcome.png", dpi=150)
    plt.close(fig)

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
