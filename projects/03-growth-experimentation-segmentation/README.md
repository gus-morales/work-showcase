# Growth Experimentation & Segmentation

Four growth analyses for a synthetic Mexican BNPL fintech: an A/B test on a repayment-reminder redesign with a proper power analysis and CUPED variance reduction, a difference-in-differences read on a regional feature rollout that wasn't randomized, RFM customer segmentation, and light NLP topic modeling on support tickets. Built on synthetic data, mirroring the experimentation and lifecycle-analytics work that sits alongside credit risk and growth analysis in fintech.

**For the full technical walkthrough (power analysis, fixed-effects regression, clustering, NMF), see the [notebook](notebooks/03_growth_experimentation_segmentation.ipynb).** This README is the short version.

> All data here is synthetically generated. No proprietary data, models, or results from any employer are used or implied. This is the same fictional company as projects 01 and 02, viewed from the growth/experimentation side.

**Skills demonstrated:** experiment design and power analysis, two-proportion hypothesis testing, CUPED variance reduction, difference-in-differences with fixed effects and a parallel-trends check, KMeans clustering for RFM segmentation, light NLP topic modeling (TF-IDF + NMF).

## The problem

Growth teams run experiments and read metrics on populations that were rarely handed to them cleanly randomized or evenly behaved. A test needs to be sized correctly before it runs, a rollout that skipped randomization still needs an honest causal read, and a customer base or a support queue needs to be broken into groups that are actually useful to act on.

## 1. A/B test: repayment-reminder redesign

A redesigned in-app reminder (clearer due date, one-tap repayment link) was tested against the existing one. The test was sized to detect a 3.5 percentage-point lift at 80% power (requiring about 2,900 users per arm) and ran on roughly 20,000 users.

| | |
|---|---|
| On-time conversion, control vs. treatment | 33.7% vs. 38.1% |
| Absolute lift | +4.4pp (95% CI: 3.1 to 5.8pp), p < 0.0001 |
| CUPED confidence interval narrowing | 11%, using pre-period revenue as the covariate |

![A/B test result](reports/figures/ab_conversion_result.png)

![CUPED variance reduction](reports/figures/cuped_variance_reduction.png)

## 2. Difference-in-differences: regional rollout

A new in-app collections feature was rolled out to 20 of 40 regions first, based on business priority rather than random assignment, which rules out a simple before/after read. A region and day fixed-effects regression, checked against a pre-period parallel-trends test first, isolates the treatment effect from any shared time trend.

| | |
|---|---|
| Pre-period trend difference (placebo check) | Not significant (p = 0.69), supports parallel trends |
| DiD estimate | +4.3pp on-time repayment (95% CI: 4.0 to 4.6pp), p < 0.0001 |

![Parallel trends](reports/figures/did_parallel_trends.png)

## 3. RFM customer segmentation

Recency, frequency, and monetary value, clustered with KMeans (k chosen by silhouette score, not fixed in advance) into three segments.

| Segment | % of customers | % of revenue |
|---|---|---|
| Champions | 16% | 46% |
| Loyal | 52% | 48% |
| Dormant | 32% | 6% |

![Segment revenue share](reports/figures/segment_revenue_share.png)

## 4. Support ticket topic modeling

TF-IDF + NMF on 1,500 synthetic support tickets recovers five topics from text alone, validated at 85% purity against the known ground-truth categories (a check only possible because the data is synthetic).

![Topic volume](reports/figures/topic_volume.png)

One topic, general account questions, doesn't cluster cleanly on its own; it scatters across the other four because it shares vocabulary with them rather than having a distinct signature. That's a real limitation of unsupervised topic modeling on short text, not something worth glossing over.

## Recommendation

Ship the reminder redesign; the lift is well outside noise and confirmed two ways (a standard test and a CUPED-adjusted one with a tighter interval). For the regional rollout, the fixed-effects estimate and the pre-period placebo check both support treating the +4.3pp effect as real rather than a pre-existing regional difference, which makes the case for extending the rollout to the remaining regions. For lifecycle marketing, the RFM segments show where the leverage actually is: roughly a third of customers are Dormant and contribute barely 6% of revenue, so a differentiated win-back offer for that group would cost little in forgone Champion/Loyal attention. For support operations, route the four cleanly-separated topics to a keyword/topic-based triage rule, but keep a human or supervised classifier in the loop for general account questions, since that category doesn't have a clean unsupervised signature to route on.

## Repo layout

- `notebooks/03_growth_experimentation_segmentation.ipynb`: full technical walkthrough, executed with all charts and results inline.
- `src/`: the reproducible pipeline (data generation, experiment design/CUPED, causal inference, segmentation, topic modeling) as standalone scripts.
- `reports/`: generated charts and CSV outputs.

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/experiment_design.py
python src/causal_inference.py
python src/segmentation.py
python src/ticket_topics.py
```

`data/` and `reports/*.csv` are gitignored; regenerate them by running the scripts above.
