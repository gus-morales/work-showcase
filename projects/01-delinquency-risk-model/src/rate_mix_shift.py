"""
Rate-mix shift decomposition: is the delinquency-rate increase during the
monitored window (months 22-24, the macro shock) explained by the
portfolio's composition shifting toward riskier segments (mix effect), or
by loans within the same segment getting riskier (rate effect)?

This quantifies the qualitative finding already in monitor_drift.py
(PSI on individual features is clean, but the model under-predicts risk
through the shock) rather than duplicating it: it checks a specific
alternative explanation, that the portfolio quietly shifted into riskier
segments, and shows how much of the delinquency increase that explanation
can and can't account for.

Uses a symmetric ("average weights, average rates") bridge decomposition,
which sums to the observed change exactly with no unexplained residual:

    mix_effect(segment)  = (w_b - w_a) * (r_a + r_b) / 2
    rate_effect(segment) = (w_a + w_b) / 2 * (r_b - r_a)

where w = segment's share of loans, r = segment's delinquency rate, in
period A (reference) vs. period B (monitored).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_RED, GREY

BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

REFERENCE_END_MONTH = 21  # same reference/monitored split as monitor_drift.py
PRIMARY_SEGMENT = "employment_type"
ROBUSTNESS_SEGMENTS = ["city_tier", "merchant_category", "acquisition_channel"]


def decompose_rate_mix_shift(weights_a: pd.Series, rates_a: pd.Series,
                              weights_b: pd.Series, rates_b: pd.Series) -> dict:
    """Pure computation: no plotting, no I/O. Segment weights should each
    sum to 1 within their period; segments missing from one period are
    treated as zero-weight in that period."""
    segments = sorted(set(weights_a.index) | set(weights_b.index))
    w_a = weights_a.reindex(segments, fill_value=0.0)
    w_b = weights_b.reindex(segments, fill_value=0.0)
    r_a = rates_a.reindex(segments, fill_value=0.0)
    r_b = rates_b.reindex(segments, fill_value=0.0)

    mix_effect = (w_b - w_a) * (r_a + r_b) / 2
    rate_effect = (w_a + w_b) / 2 * (r_b - r_a)
    total_delta = float((w_b * r_b).sum() - (w_a * r_a).sum())

    return {
        "segments": segments,
        "mix_effect": mix_effect,
        "rate_effect": rate_effect,
        "total_delta": total_delta,
        "mix_total": float(mix_effect.sum()),
        "rate_total": float(rate_effect.sum()),
    }


def _segment_weights_and_rates(df, segment_col):
    weights = df.groupby(segment_col).size() / len(df)
    rates = df.groupby(segment_col)["delinquent_30dpd"].mean()
    return weights, rates


def rate_mix_shift(df, segment_col, source_note, plot=True):
    reference = df[df["origination_month"] <= REFERENCE_END_MONTH]
    monitored = df[df["origination_month"] > REFERENCE_END_MONTH]

    w_a, r_a = _segment_weights_and_rates(reference, segment_col)
    w_b, r_b = _segment_weights_and_rates(monitored, segment_col)
    result = decompose_rate_mix_shift(w_a, r_a, w_b, r_b)

    mix_share = result["mix_total"] / result["total_delta"]
    rate_share = result["rate_total"] / result["total_delta"]

    print(f"[{segment_col}] Overall delinquency-rate change: {result['total_delta']:+.3%}")
    print(f"[{segment_col}] Mix effect: {result['mix_total']:+.3%} ({mix_share:.1%} of the change)")
    print(f"[{segment_col}] Rate effect: {result['rate_total']:+.3%} ({rate_share:.1%} of the change)")

    if not plot:
        return result

    # Chart 1: total change split into mix vs. rate effect.
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ["Mix effect\n(composition shift)", "Rate effect\n(within-segment)"]
    vals = [result["mix_total"] * 100, result["rate_total"] * 100]
    colors = [GREY, MUTED_RED]
    ax.bar(bars, vals, color=colors, width=0.5, zorder=3)
    ax.axhline(0, color=GREY, linewidth=1)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.05 if v >= 0 else -0.15), f"{v:+.2f}pp", ha="center", fontsize=10.5, color="#333")
    style_ax(ax, title=f"{rate_share:.0%} of the delinquency-rate increase is a rate effect, not a mix shift",
             subtitle=f"Decomposed by {segment_col.replace('_', ' ')}, months 1-{REFERENCE_END_MONTH} vs. {REFERENCE_END_MONTH+1}-24",
             ylabel="Contribution to rate change (percentage points)")
    savefig(fig, FIG_DIR / "rate_mix_shift_decomposition.png", footnote=source_note)

    # Chart 2: per-segment rate before vs. after, showing which segments moved most.
    segs = result["segments"]
    labels = [s.replace("_", " ") for s in segs]
    x = np.arange(len(segs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, r_a.reindex(segs, fill_value=0) * 100, width, color=SLATE, label="Reference (1-21)", zorder=3)
    ax.bar(x + width / 2, r_b.reindex(segs, fill_value=0) * 100, width, color=MUTED_TEAL, label=f"Monitored ({REFERENCE_END_MONTH+1}-24)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    style_ax(ax, title=f"Every {segment_col.replace('_', ' ')} segment got riskier, not just one",
             subtitle="Delinquency rate by segment, reference vs. monitored window",
             ylabel="Delinquency rate (%)")
    ax.legend(fontsize=9.5, loc="upper left")
    savefig(fig, FIG_DIR / "rate_mix_shift_by_segment.png", footnote=source_note)

    return result


def main():
    df = pd.read_csv(BASE / "data" / "loans.csv")
    n_ref = (df["origination_month"] <= REFERENCE_END_MONTH).sum()
    n_mon = (df["origination_month"] > REFERENCE_END_MONTH).sum()
    source_note = f"Source: synthetic BNPL loan data · reference n = {n_ref:,}, monitored n = {n_mon:,} loans"

    rate_mix_shift(df, PRIMARY_SEGMENT, source_note)

    print("\nRobustness check across other segment dimensions (rate-effect share should stay dominant):")
    for seg in ROBUSTNESS_SEGMENTS:
        w_a, r_a = _segment_weights_and_rates(df[df["origination_month"] <= REFERENCE_END_MONTH], seg)
        w_b, r_b = _segment_weights_and_rates(df[df["origination_month"] > REFERENCE_END_MONTH], seg)
        result = decompose_rate_mix_shift(w_a, r_a, w_b, r_b)
        print(f"  {seg}: rate effect = {result['rate_total']/result['total_delta']:.1%} of the change")

    print("\nWrote reports/figures/rate_mix_shift_decomposition.png, rate_mix_shift_by_segment.png")


if __name__ == "__main__":
    main()
