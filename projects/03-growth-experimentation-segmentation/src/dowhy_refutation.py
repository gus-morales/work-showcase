"""
Model / identify / estimate / refute: DoWhy's causal-inference workflow
applied to the same regional-rollout DiD as causal_inference.py, adding
a generic refutation suite that checks the estimate from angles the
DiD-specific checks in robustness_checks.py (wild cluster bootstrap,
Honest DiD) don't cover.

The estimator here (backdoor.linear_regression, region and day as the
adjustment set) mirrors the region+day fixed-effects structure of the
hand-rolled regression in causal_inference.py, but unweighted, so the
point estimate lands close to the simple 2x2 DiD there rather than the
weighted estimate; the two are independent cross-checks, not
duplicates of the same computation.

Three refuters:
1. random_common_cause: adds a random confounder. If the estimate
   changes much, real unmeasured confounders would be dangerous too.
2. placebo_treatment_refuter: replaces the real treatment with a
   permuted one. DoWhy's significance test for this refuter checks
   whether *zero* falls comfortably inside the distribution of
   placebo-simulated effects, not whether the original estimate does,
   so a *non-significant* result is the desired outcome here: it means
   the placebo effects cluster around zero, as a placebo should.
3. data_subset_refuter: refits on random 80% subsets. If the estimate
   is unstable under subsampling, it isn't being driven consistently
   by the whole dataset.

DoWhy 0.14 note: data_subset_refuter's refute_data_subset does not
convert an integer random_state into an np.random.RandomState before
threading it into pandas' .sample(random_state=...) inside its
per-simulation loop, unlike the other two refuters, which do this
conversion at the top of their own refute_* functions. Passing a plain
int here silently reruns the identical subset on every simulation
(verified directly: 100 "simulations" with random_state=13 returned
the same value 100 times, zero variance, and a degenerate p-value).
Passing an actual RandomState object instead works correctly, since
pandas advances that object's stream across calls. Every refuter call
below is passed an explicit np.random.RandomState because it is
required for correctness here, independent of any style preference.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dowhy import CausalModel

from style import set_style, style_ax, savefig, SLATE, MUTED_TEAL, MUTED_RED, GREY, INK

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
set_style()

SEED = 13


def build_and_estimate(df):
    """region_id and day are cast to strings so DoWhy dummy-encodes
    them as categorical adjustment variables, the same fixed-effects
    structure as C(region_id) + C(day) in causal_inference.py."""
    d = df.copy()
    d["treated_flag"] = (d.group == "treated").astype(int)
    d["post_flag"] = d.post_rollout.astype(int)
    d["did_treatment"] = d["treated_flag"] * d["post_flag"]
    d["region_id"] = d["region_id"].astype(str)
    d["day"] = d["day"].astype(str)

    model = CausalModel(
        data=d, treatment="did_treatment", outcome="on_time_rate",
        common_causes=["region_id", "day"],
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    return model, identified_estimand, estimate


def run_refutation_suite(model, identified_estimand, estimate, seed=SEED, num_simulations=30):
    r1 = model.refute_estimate(
        identified_estimand, estimate, method_name="random_common_cause",
        num_simulations=num_simulations, random_state=np.random.RandomState(seed),
    )
    r2 = model.refute_estimate(
        identified_estimand, estimate, method_name="placebo_treatment_refuter",
        num_simulations=num_simulations, placebo_type="permute", random_state=np.random.RandomState(seed),
    )
    r3 = model.refute_estimate(
        identified_estimand, estimate, method_name="data_subset_refuter",
        subset_fraction=0.8, num_simulations=num_simulations, random_state=np.random.RandomState(seed),
    )
    return {
        "random_common_cause": {"new_effect": r1.new_effect, **r1.refutation_result},
        "placebo_treatment": {"new_effect": r2.new_effect, **r2.refutation_result},
        "data_subset": {"new_effect": r3.new_effect, **r3.refutation_result},
    }


def refutation_chart(estimate_value, results, source_note):
    labels = ["Original\nestimate", "Random common\ncause added", "Placebo\ntreatment", "80% data\nsubset"]
    values = [
        estimate_value, results["random_common_cause"]["new_effect"],
        results["placebo_treatment"]["new_effect"], results["data_subset"]["new_effect"],
    ]
    colors = [SLATE, MUTED_TEAL, MUTED_RED, MUTED_TEAL]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(labels, [v * 100 for v in values], color=colors, width=0.55, zorder=3)
    for i, v in enumerate(values):
        y = v * 100
        label_y = y + 0.15 if y >= -0.1 else y - 0.4
        ax.text(i, label_y, f"{y:.2f}pp", ha="center", fontsize=10, color=INK)
    ax.axhline(0, color=GREY, linewidth=1)
    style_ax(ax, title="The DiD estimate survives three refutation checks",
             subtitle="DoWhy refutation suite: original estimate vs. each refuter's re-estimated effect",
             ylabel="On-time repayment effect (pp)")
    savefig(fig, FIG_DIR / "dowhy_refutation.png", footnote=source_note)


def dowhy_refutation(df, source_note):
    model, identified_estimand, estimate = build_and_estimate(df)
    results = run_refutation_suite(model, identified_estimand, estimate)

    print(f"DoWhy backdoor.linear_regression estimate: {estimate.value*100:.2f}pp")
    for name, r in results.items():
        outcome = "FAILS" if r["is_statistically_significant"] else "passes"
        print(f"{name}: new_effect={r['new_effect']*100:.2f}pp, p={r['p_value']:.2f}, {outcome} refutation check")

    refutation_chart(estimate.value, results, source_note)
    return {"estimate": estimate.value, "refutation_results": results}


def main():
    df = pd.read_csv(DATA_DIR / "regional_rollout.csv")
    n_regions = df.region_id.nunique()
    source_note = f"Source: synthetic BNPL regional panel · {n_regions} regions, {df.day.nunique()} days"
    dowhy_refutation(df, source_note)
    print("Wrote reports/figures/dowhy_refutation.png")


if __name__ == "__main__":
    main()
