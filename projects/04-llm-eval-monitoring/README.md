# Evaluating & Monitoring an LLM-Powered Feature

A generic company ships an LLM-drafted reply feature for its support inbox: the model drafts a response, and it's either auto-sent or routed to a human reviewer depending on risk. This project covers the data-science layer around that feature, not the feature itself: validating the automated judge used to score reply quality, A/B testing a prompt change, monitoring quality in production, and picking a cost-based auto-send threshold. Built on synthetic data, mirroring the AI-ops/eval work that sits alongside classical modeling as more teams ship LLM-powered features.

**For the full technical walkthrough (kappa validation, power analysis, control charts, cost curves), see the [notebook](notebooks/04_llm_eval_monitoring.ipynb).** This README is the short version.

> All data here is synthetically generated. No proprietary data, models, or results from any employer are used or implied. Deliberately not tied to the fictional company in projects 01-03; this methodology is meant to generalize across any LLM-powered feature, not read as a one-company demo.

**Skills and tools featured:**

- LLM-judge validation against human labels (Cohen's kappa, bias decomposition)
- Experiment design and two-proportion A/B testing
- Statistical process control (p-charts) for output-quality monitoring
- Cost-based decision threshold optimization

## The problem

Teams shipping LLM-powered features tend to reach for an LLM-judge to score output quality because human review of everything doesn't scale, then trust the judge's numbers without checking whether it's actually a reliable stand-in for a human. Meanwhile the usual production-monitoring playbook (checking whether input features have drifted) doesn't catch a regression in what the model outputs, and the auto-send/review-queue split is usually set by gut feel rather than by what each type of mistake actually costs.

## 1. Validating the LLM-judge against human labels

A 600-ticket golden set was scored by both a human rater and the LLM-judge on a 1-5 scale, before trusting the judge for anything downstream.

| | |
|---|---|
| Exact agreement | 52.7% |
| Within-1-point agreement | 96.7% |
| Quadratic-weighted kappa | 0.49 (below the 0.60 "substantial agreement" floor) |
| Judge bias vs. human | +0.31 points, systematically generous (p < 0.0001) |

![Judge confusion matrix](reports/figures/judge_confusion_matrix.png)

The bias concentrates on complaints, the category where a bad reply does the most damage and where catching it matters most.

![Judge bias by category](reports/figures/judge_bias_by_category.png)

## 2. A/B test: a revised drafting prompt

Prompt v2 adds an explicit instruction to acknowledge the issue and give one concrete next step, tested against the baseline prompt on ~6,000 tickets (well above the ~1,200/arm the power analysis called for at a 4pp minimum detectable effect).

| | |
|---|---|
| Judge-acceptable rate, v1 vs. v2 | 84.0% vs. 90.2% |
| Absolute lift | +6.2pp (95% CI: 4.5 to 7.9pp), p < 0.0001 |

![A/B test result](reports/figures/ab_test_result.png)

Since the judge's generosity bias applies to both arms about equally, this relative lift is a fair read even though neither arm's raw acceptable-rate should be quoted as a trustworthy standalone number.

## 3. Production monitoring: catching a silent quality regression

A daily p-chart (control chart for a proportion) tracks the judge-scored acceptable rate against control limits set from a stable reference period. A regression was injected on day 90 with no change in ticket volume or category mix, exactly the kind of shift that standard input-drift monitoring would miss entirely.

| | |
|---|---|
| Reference-period center line | 82.6% acceptable |
| Regression detected | Day 90, the same day it started (3-day run rule) |

![Quality control chart](reports/figures/quality_control_chart.png)

## 4. Guardrail threshold: auto-send vs. route-to-human

A lightweight safety classifier (AUC 0.85) scores every drafted reply before it's sent. The threshold was set from actual costs rather than a default 0.5 cutoff: a bad reply that gets auto-sent costs far more (remediation, trust damage) than a fine reply that gets routed to a human anyway (reviewer time), but that reviewer-time cost is paid on every routed reply, good or bad.

| | |
|---|---|
| Cost-optimal threshold | 0.16, vs. a naive 0.50 |
| Expected cost reduction | 48.3% ($13,730 vs. $26,556 per 8,000 replies) |
| Share routed to human review | 72.7% at the optimal threshold vs. 27.2% at 0.50 |

![Guardrail cost curve](reports/figures/guardrail_cost_curve.png)

## Recommendation

Don't retire human review of the judge itself: a kappa of 0.49 is below the standard bar for substantial agreement, and the miscalibration concentrates on complaints, the category where it's most costly, so those should get more frequent human audits than the aggregate agreement number alone would suggest. The prompt change is a clear ship: a 6.2pp lift with a tight interval, and the relative-comparison logic holds even given the judge's known bias. For monitoring, the control chart is doing its job (same-day detection here), so the main follow-up is deciding the run-rule length against how expensive a slow-to-detect regression would actually be. For the guardrail, the threshold itself is already cost-optimal; the constraint is the classifier's precision at AUC 0.85, and improving that would do more to shrink the review queue than adjusting the threshold further.

## Repo layout

- `notebooks/04_llm_eval_monitoring.ipynb`: full technical walkthrough, executed with all charts and results inline.
- `src/`: the reproducible pipeline (data generation, judge validation, A/B test, drift monitoring, guardrail threshold) as standalone scripts.
- `tests/`: pytest suite covering data-generation invariants and the pure computation behind the kappa/bias calculation, the A/B result, the control-chart detection logic, and the cost-optimal threshold search.
- `reports/`: generated charts.

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/eval_framework.py
python src/ab_test.py
python src/drift_monitoring.py
python src/guardrail_threshold.py
```

`data/` is gitignored; regenerate it by running the scripts above.

## Tests

```bash
pytest tests/ -v
```

Runs in CI on every push (see the badge at the [repo root](../../README.md)).
