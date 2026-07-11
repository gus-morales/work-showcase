"""
Synthetic data for project 04: evaluating and monitoring an LLM-drafted
support-reply feature. Domain-agnostic on purpose (not tied to the BNPL
company in projects 01-03) since this is meant to read as a reusable
methodology, not a feature demo for one company.

Four datasets, each feeding one part of the analysis:

1. golden_eval_set.csv       - a labeled set of (request, drafted reply)
   pairs, each scored by both a human rater and an LLM-judge, used to
   validate whether the automated judge can be trusted as a stand-in for
   human review.
2. ab_test_results.csv       - two reply-drafting prompt versions, judged
   pass/fail by the (now-validated) automated judge, for a standard
   two-proportion A/B test.
3. quality_monitoring.csv    - a daily time series of judge-scored replies,
   including a silent quality regression partway through, for production
   monitoring.
4. guardrail_scores.csv      - a safety-classifier risk score plus a
   ground-truth "problematic reply" label, for cost-based auto-send vs.
   route-to-human threshold selection.

Run:
    python src/generate_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 7
rng = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

CATEGORIES = [
    "billing_question", "technical_issue", "cancellation_request",
    "account_access", "how_to_question", "complaint",
]
# Latent difficulty of drafting a genuinely good reply, by category.
# Complaints are hardest: emotionally charged, harder to fully resolve
# in a single reply, easiest for a lenient judge to rate too generously.
CATEGORY_QUALITY_MEAN = {
    "billing_question": 0.78,
    "technical_issue": 0.65,
    "cancellation_request": 0.70,
    "account_access": 0.80,
    "how_to_question": 0.85,
    "complaint": 0.55,
}
CATEGORY_WEIGHTS = [0.20, 0.20, 0.14, 0.16, 0.16, 0.14]


def _true_quality(n, categories):
    """Latent, unobserved 'actual' reply quality in [0, 1], the thing both
    the human rater and the LLM-judge are each independently, imperfectly,
    trying to measure."""
    means = np.array([CATEGORY_QUALITY_MEAN[c] for c in categories])
    return np.clip(rng.normal(means, 0.12), 0.03, 0.99)


def _to_likert(quality, bias=0.0, noise_sd=0.35, rng_local=None):
    r = rng_local if rng_local is not None else rng
    raw = quality * 4 + 1 + bias + r.normal(0, noise_sd, size=len(quality))
    return np.clip(np.round(raw), 1, 5).astype(int)


# ---------------------------------------------------------------------
# 1. Golden eval set: human label vs. LLM-judge label
# ---------------------------------------------------------------------

def make_golden_eval_set(n=600):
    categories = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)
    quality = _true_quality(n, categories)

    # Human rater: roughly unbiased, moderate noise.
    human_label = _to_likert(quality, bias=0.0, noise_sd=0.35)

    # LLM-judge: systematically generous (a well-documented failure mode of
    # using an LLM to grade LLM output), and specifically most generous on
    # the hardest category (complaints), which is exactly where a lenient
    # judge does the most damage - it hides the cases most worth catching.
    judge_bias = np.where(categories == "complaint", 0.55, 0.30)
    judge_label = _to_likert(quality, bias=judge_bias, noise_sd=0.45)

    return pd.DataFrame({
        "ticket_id": np.arange(1, n + 1),
        "category": categories,
        "human_label": human_label,
        "judge_label": judge_label,
        "human_acceptable": (human_label >= 4).astype(int),
        "judge_acceptable": (judge_label >= 4).astype(int),
    })


# ---------------------------------------------------------------------
# 2. A/B test: two reply-drafting prompt versions
# ---------------------------------------------------------------------

def make_ab_test_results(n=6000):
    """Prompt v2 adds an explicit instruction to acknowledge the issue and
    give one concrete next step. Judged with the same generous-judge model
    as the golden set; the absolute acceptable-rate here is inflated by
    that bias, but the bias applies equally to both arms; so the *relative*
    lift between arms is still a valid read, even though the absolute rate
    isn't a trustworthy standalone number."""
    categories = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)
    arm = rng.choice(["v1_baseline", "v2_revised"], size=n, p=[0.5, 0.5])

    quality = _true_quality(n, categories)
    true_lift = np.where(arm == "v2_revised", 0.06, 0.0)  # genuine underlying improvement
    quality = np.clip(quality + true_lift, 0.03, 0.99)

    judge_bias = np.where(categories == "complaint", 0.55, 0.30)
    judge_label = _to_likert(quality, bias=judge_bias, noise_sd=0.45)

    return pd.DataFrame({
        "ticket_id": np.arange(1, n + 1),
        "category": categories,
        "arm": arm,
        "judge_label": judge_label,
        "judge_acceptable": (judge_label >= 4).astype(int),
    })


# ---------------------------------------------------------------------
# 3. Quality monitoring: daily judge-scored acceptable rate
# ---------------------------------------------------------------------

def make_quality_monitoring(n_days=120, regression_day=90, avg_tickets_per_day=80):
    """A silent upstream change (e.g. a provider model swap or a prompt
    template regression) drops reply quality starting on `regression_day`.
    No input feature changes; only the output quality does, so this is
    only catchable by scoring outputs, not by monitoring request volume
    or category mix."""
    rows = []
    for day in range(n_days):
        n_today = rng.poisson(avg_tickets_per_day)
        categories = rng.choice(CATEGORIES, size=n_today, p=CATEGORY_WEIGHTS)
        quality = _true_quality(n_today, categories)
        if day >= regression_day:
            quality = np.clip(quality - 0.22, 0.02, 0.99)
        judge_bias = np.where(categories == "complaint", 0.55, 0.30)
        judge_label = _to_likert(quality, bias=judge_bias, noise_sd=0.45)
        acceptable = (judge_label >= 4).astype(int)
        rows.append((day, n_today, int(acceptable.sum()), acceptable.mean()))

    return pd.DataFrame(rows, columns=["day", "n_tickets", "n_acceptable", "acceptable_rate"])


# ---------------------------------------------------------------------
# 4. Guardrail scores: safety-classifier risk vs. ground-truth bad reply
# ---------------------------------------------------------------------

def make_guardrail_scores(n=8000, bad_rate=0.08):
    """A separate lightweight safety classifier scores every drafted reply
    with a risk probability before it's either auto-sent or routed to a
    human. true_bad is the (only-knowable-in-hindsight) ground truth: did
    the reply contain a hallucinated fact, a wrong policy statement, or an
    inappropriate tone. The classifier is decent but imperfect (AUC ~0.85),
    which is realistic and is the whole reason a threshold decision
    matters."""
    true_bad = rng.binomial(1, bad_rate, size=n)

    # Risk score: bad replies get a higher-mean score, good replies a
    # lower-mean one, both on a logit scale then squashed to [0, 1], which
    # produces a classifier with imperfect but real separation (AUC ~0.85,
    # not a suspiciously perfect one).
    logit = np.where(true_bad == 1, rng.normal(0.95, 1.3, size=n), rng.normal(-0.95, 1.3, size=n))
    risk_score = 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        "reply_id": np.arange(1, n + 1),
        "risk_score": risk_score.round(4),
        "true_bad": true_bad,
    })


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    golden = make_golden_eval_set()
    golden.to_csv(OUT_DIR / "golden_eval_set.csv", index=False)
    print(f"Wrote {len(golden):,} golden eval rows -> data/golden_eval_set.csv")
    print(f"Human acceptable rate: {golden['human_acceptable'].mean():.1%}  "
          f"Judge acceptable rate: {golden['judge_acceptable'].mean():.1%}")

    ab = make_ab_test_results()
    ab.to_csv(OUT_DIR / "ab_test_results.csv", index=False)
    print(f"Wrote {len(ab):,} A/B test rows -> data/ab_test_results.csv")
    print(ab.groupby("arm")["judge_acceptable"].mean())

    monitoring = make_quality_monitoring()
    monitoring.to_csv(OUT_DIR / "quality_monitoring.csv", index=False)
    print(f"Wrote {len(monitoring):,} daily rows -> data/quality_monitoring.csv")

    guardrail = make_guardrail_scores()
    guardrail.to_csv(OUT_DIR / "guardrail_scores.csv", index=False)
    print(f"Wrote {len(guardrail):,} guardrail rows -> data/guardrail_scores.csv")
    print(f"True bad rate: {guardrail['true_bad'].mean():.1%}")


if __name__ == "__main__":
    main()
