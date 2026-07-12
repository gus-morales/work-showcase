"""Exploratory analysis across all four datasets in this project, before
judge validation, the A/B test, monitoring, or guardrail thresholding:
saves charts to reports/figures and a numeric summary to
reports/eda_summary.md."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_RED

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

set_style()


def main():
    golden = pd.read_csv(DATA_DIR / "golden_eval_set.csv")
    ab = pd.read_csv(DATA_DIR / "ab_test_results.csv")
    monitoring = pd.read_csv(DATA_DIR / "quality_monitoring.csv")
    guardrail = pd.read_csv(DATA_DIR / "guardrail_scores.csv")

    # 1. Golden set: human vs. judge label distribution, before computing
    # kappa or bias formally. The judge already looks visibly more generous
    # here, stacked toward 4s and 5s where the human rater spreads more
    # evenly across 3-5.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, 6)
    width = 0.38
    human_counts = golden["human_label"].value_counts().reindex(x, fill_value=0)
    judge_counts = golden["judge_label"].value_counts().reindex(x, fill_value=0)
    ax.bar(x - width / 2, human_counts.values, width, color=SLATE, label="Human", zorder=3)
    ax.bar(x + width / 2, judge_counts.values, width, color=MUTED_RED, label="LLM-judge", zorder=3)
    ax.set_xticks(x)
    style_ax(ax, title="The judge already looks more generous than the human rater",
             subtitle="Score distribution (1-5 scale), human vs. LLM-judge, same replies",
             xlabel="Score", ylabel="Count")
    ax.legend(loc="upper left")
    savefig(fig, FIG_DIR / "golden_score_distribution.png",
            footnote=f"Source: synthetic golden eval set (src/generate_data.py) · n = {len(golden):,} replies")

    # 2. A/B test: category mix balance between arms. A skewed mix would
    # confound the arm comparison with a category-difficulty difference
    # instead of a real prompt-version effect.
    mix = pd.crosstab(ab["arm"], ab["category"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(mix.shape[1])
    width = 0.38
    ax.bar(x - width / 2, mix.loc["v1_baseline"].values, width, color=SLATE, label="v1 (baseline)", zorder=3)
    ax.bar(x + width / 2, mix.loc["v2_revised"].values, width, color=MUTED_TEAL, label="v2 (revised)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in mix.columns], rotation=20, ha="right")
    style_ax(ax, title="Ticket category mix is balanced across arms",
             subtitle="Share of tickets by category, v1 vs. v2",
             ylabel="Share of tickets (%)")
    ax.set_ylim(0, mix.values.max() * 1.28)
    ax.legend(loc="upper center", ncol=2, fontsize=9.5)
    savefig(fig, FIG_DIR / "ab_category_balance.png",
            footnote=f"Source: synthetic A/B test data (src/generate_data.py) · n = {len(ab):,} tickets")

    # 3. Monitoring: daily ticket volume over the observation window. Flat
    # before and after the quality regression, confirming the later
    # acceptable-rate drop is a quality change, not a volume artifact.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(monitoring["day"], monitoring["n_tickets"], color=SLATE, linewidth=1.4)
    ax.axvline(90, ls="--", color=MUTED_RED, linewidth=1.2, label="Quality regression starts")
    style_ax(ax, title="Ticket volume stays flat across the regression window",
             subtitle="Daily ticket count, full 120-day monitoring period",
             xlabel="Day", ylabel="Tickets")
    ax.legend(loc="upper left")
    savefig(fig, FIG_DIR / "monitoring_volume.png",
            footnote=f"Source: synthetic monitoring data (src/generate_data.py) · n = {len(monitoring)} days")

    # 4. Guardrail: raw risk-score separation by ground truth, before any
    # threshold gets picked.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=guardrail, x="risk_score", hue="true_bad", fill=True, common_norm=False,
                palette=[MUTED_TEAL, MUTED_RED], alpha=0.35, linewidth=1.3, ax=ax, legend=False)
    ax.plot([], [], color=MUTED_TEAL, linewidth=6, alpha=0.35, label="Genuine reply")
    ax.plot([], [], color=MUTED_RED, linewidth=6, alpha=0.35, label="Problematic reply")
    ax.legend(loc="upper right")
    style_ax(ax, title="The safety classifier separates the two groups, but imperfectly",
             subtitle="Risk score distribution by ground-truth label",
             xlabel="Risk score", ylabel="Density")
    savefig(fig, FIG_DIR / "guardrail_score_distribution.png",
            footnote=f"Source: synthetic guardrail data (src/generate_data.py) · n = {len(guardrail):,} replies")

    # Summary markdown
    lines = ["# EDA summary\n"]
    lines.append(f"- Golden eval set: {len(golden):,} replies, human mean score "
                 f"{golden['human_label'].mean():.2f}, judge mean score {golden['judge_label'].mean():.2f}\n")
    lines.append(f"- A/B test: {len(ab):,} tickets, "
                 f"{(ab['arm'] == 'v1_baseline').sum():,} v1 vs. {(ab['arm'] == 'v2_revised').sum():,} v2, "
                 f"category mix balanced within a couple points across arms\n")
    lines.append(f"- Monitoring: {len(monitoring)} days, {monitoring['n_tickets'].mean():.1f} tickets/day on "
                 f"average, {monitoring[monitoring['day'] < 90]['n_tickets'].mean():.1f} pre-regression vs. "
                 f"{monitoring[monitoring['day'] >= 90]['n_tickets'].mean():.1f} post-regression\n")
    lines.append(f"- Guardrail: {len(guardrail):,} replies, {guardrail['true_bad'].mean():.1%} true bad rate, "
                 f"median risk score {guardrail[guardrail['true_bad'] == 0]['risk_score'].median():.2f} (genuine) "
                 f"vs. {guardrail[guardrail['true_bad'] == 1]['risk_score'].median():.2f} (problematic)\n")
    (BASE / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("EDA complete. Figures + summary written to reports/.")


if __name__ == "__main__":
    main()
