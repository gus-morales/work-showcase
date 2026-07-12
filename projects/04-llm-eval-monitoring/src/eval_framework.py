"""
Validates the LLM-judge against human labels on a golden eval set, before
trusting the judge for anything downstream (the A/B test and the
monitoring pipeline both rely on it). An automated judge that looks
reliable in aggregate can still be systematically miscalibrated on the
subset that matters most, which is the specific thing checked here,
rather than a plain "does the judge roughly agree with humans" average.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from style import set_style, style_ax, savefig, SLATE, MUTED_RED, GREY, HEATMAP_CMAP, HEATMAP_TEXT_LOW, HEATMAP_TEXT_HIGH

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

ACCEPTABLE_AGREEMENT_KAPPA = 0.60  # commonly cited floor for "substantial" agreement


def compute_judge_validation(df: pd.DataFrame) -> dict:
    """Pure computation: no plotting, no file I/O, so this is directly
    unit-testable on a small hand-built frame."""
    human = df["human_label"].values
    judge = df["judge_label"].values

    exact_agreement = float(np.mean(human == judge))
    adjacent_agreement = float(np.mean(np.abs(human - judge) <= 1))
    kappa = float(cohen_kappa_score(human, judge, weights="quadratic"))
    corr = float(np.corrcoef(human, judge)[0, 1])

    diff = judge - human
    bias_overall = float(diff.mean())
    t_stat, p_value = stats.ttest_1samp(diff, 0)

    bias_by_category = (
        df.assign(diff=diff)
        .groupby("category")["diff"]
        .mean()
        .sort_values(ascending=False)
    )

    return {
        "exact_agreement_rate": exact_agreement,
        "adjacent_agreement_rate": adjacent_agreement,
        "kappa": kappa,
        "correlation": corr,
        "bias_overall": bias_overall,
        "bias_p_value": float(p_value),
        "bias_by_category": bias_by_category,
    }


def judge_validation(df, source_note):
    result = compute_judge_validation(df)

    # Confusion matrix: human label (rows) vs. judge label (columns).
    labels = [1, 2, 3, 4, 5]
    mat = pd.crosstab(df["human_label"], df["judge_label"]).reindex(index=labels, columns=labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mat.values, cmap=HEATMAP_CMAP, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            v = mat.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9.5,
                    color=HEATMAP_TEXT_HIGH if v > mat.values.max() * 0.5 else HEATMAP_TEXT_LOW)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    style_ax(ax, title=f"Judge runs generous: kappa={result['kappa']:.2f}, "
                        f"exact agreement={result['exact_agreement_rate']:.0%}",
             subtitle="Human label (rows) vs. LLM-judge label (columns), 1-5 scale",
             xlabel="Judge label", ylabel="Human label", grid_axis=None)
    savefig(fig, FIG_DIR / "judge_confusion_matrix.png", footnote=source_note)

    bias = result["bias_by_category"]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [MUTED_RED if v == bias.max() else SLATE for v in bias.values]
    ax.barh([c.replace("_", " ") for c in bias.index], bias.values, color=colors, zorder=3, height=0.55)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.invert_yaxis()
    style_ax(ax, title="The judge is most generous on complaints, the hardest category",
             subtitle="Mean (judge label - human label) by ticket category",
             xlabel="Judge minus human (Likert points)", grid_axis="x")
    savefig(fig, FIG_DIR / "judge_bias_by_category.png", footnote=source_note)

    print(f"Exact agreement: {result['exact_agreement_rate']:.1%}  "
          f"Within-1 agreement: {result['adjacent_agreement_rate']:.1%}")
    print(f"Quadratic-weighted kappa: {result['kappa']:.3f} "
          f"({'meets' if result['kappa'] >= ACCEPTABLE_AGREEMENT_KAPPA else 'below'} "
          f"the {ACCEPTABLE_AGREEMENT_KAPPA} substantial-agreement floor)")
    print(f"Judge bias: {result['bias_overall']:+.2f} points (p={result['bias_p_value']:.5f})")
    print("Bias by category:")
    print(bias.round(2).to_string())
    return result


def main():
    df = pd.read_csv(DATA_DIR / "golden_eval_set.csv")
    source_note = f"Source: synthetic golden eval set · n = {len(df):,} labeled tickets"
    judge_validation(df, source_note)
    print("Wrote reports/figures/judge_confusion_matrix.png, judge_bias_by_category.png")


if __name__ == "__main__":
    main()
