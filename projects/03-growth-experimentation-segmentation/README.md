# Growth Experimentation & Segmentation

Five growth analyses for a synthetic BNPL fintech: an A/B test on a repayment-reminder redesign with a proper power analysis and CUPED variance reduction, an uplift/CATE model on that same test showing who actually benefits, a difference-in-differences read on a regional feature rollout that wasn't randomized, RFM customer segmentation, and light NLP topic modeling on support tickets. Built on synthetic data, mirroring the experimentation and lifecycle-analytics work that sits alongside credit risk and growth analysis in fintech.

**For the full technical walkthrough (power analysis, fixed-effects regression, uplift modeling, clustering, NMF), see the [notebook](notebooks/03_growth_experimentation_segmentation.ipynb).** This README is the short version.

> All data here is synthetically generated. No proprietary data, models, or results from any employer are used or implied. This is the same fictional company as projects 01 and 02, viewed from the growth/experimentation side.

**Skills and tools featured:**

- Experiment design and power analysis
- Two-proportion hypothesis testing
- CUPED variance reduction
- Multi-armed bandits (Thompson Sampling) vs. fixed-horizon testing
- Uplift/CATE modeling (T-learner and EconML's CausalForestDML) with Qini-curve validation
- Difference-in-differences with fixed effects and a parallel-trends check
- Wild cluster bootstrap and Honest DiD sensitivity analysis
- DoWhy's causal model/identify/estimate/refute framework
- KMeans clustering for RFM segmentation
- Light NLP topic modeling (TF-IDF + NMF)

## The problem

Growth teams run experiments and read metrics on populations that were rarely handed to them cleanly randomized or evenly behaved. A test needs to be sized correctly before it runs, a rollout that skipped randomization still needs an honest causal read, and a customer base or a support queue needs to be broken into groups that are actually useful to act on.

## 1. A/B test: repayment-reminder redesign

A redesigned in-app reminder (clearer due date, one-tap repayment link) was tested against the existing one. The test was sized to detect a 3.5 percentage-point lift at 80% power (requiring about 2,900 users per arm) and ran on roughly 40,000 users.

| | |
|---|---|
| On-time conversion, control vs. treatment | 34.3% vs. 38.5% |
| Absolute lift | +4.2pp (95% CI: 3.3 to 5.1pp), p < 0.0001 |
| CUPED confidence interval narrowing | 11%, using pre-period revenue as the covariate |

The standard test shows the lift clearly (Figure 1); a CUPED-adjusted estimate on the same data narrows the interval further (Figure 2).

![A/B test result](reports/figures/ab_conversion_result.png)

*Figure 1. On-time conversion by arm, control vs. treatment.*

![CUPED variance reduction](reports/figures/cuped_variance_reduction.png)

*Figure 2. Estimated treatment lift, standard vs. CUPED-adjusted, 95% CI.*

### Sequential experimentation: what an adaptive design would have cost or saved

The fixed 50/50 design above is the right choice when a clean, unbiased effect estimate is the goal, exactly what section 1's z-test and CUPED interval need. But every user sent to the losing arm during the test is a real cost, and a fixed design keeps paying it at a constant rate for the whole test, even once the result is obvious. A Thompson Sampling multi-armed bandit, simulated here over the same total traffic (40,000 users) and the same two realized conversion rates (34.3% vs. 38.5%), shows the size of that cost: it shifts traffic toward the treatment arm as its posterior pulls ahead (Figure 11), reaching 96.8% of traffic by the end, and banks 886 more conversions over the same traffic by capping how long it keeps exposing users to the losing arm (Figure 12), a 93.6% reduction in cumulative regret against the fixed design.

| | |
|---|---|
| Total conversions, fixed 50/50 vs. Thompson Sampling (same 40,000 users) | 14,521 vs. 15,407 |
| Additional conversions banked by Thompson Sampling | 886 |
| Cumulative regret reduction | 93.6% |
| Traffic on treatment by the end of the run | 50% (fixed) vs. 96.8% (Thompson Sampling) |

![Traffic allocation over time](reports/figures/bandit_traffic_allocation.png)

*Figure 11. Cumulative share of traffic allocated to the treatment arm, fixed 50/50 design vs. Thompson Sampling.*

![Cumulative regret](reports/figures/bandit_cumulative_regret.png)

*Figure 12. Cumulative regret (expected conversions forgone vs. always playing the better arm), fixed design vs. Thompson Sampling.*

The catch is exactly what makes the fixed design useful in the first place: because Thompson Sampling's allocation responds to outcomes as they arrive, the resulting data no longer satisfies the assumptions the standard two-proportion z-test and its confidence interval rely on. A real deployment that wanted both the bandit's lower regret and a valid end-of-test inference would need always-valid or mixture-sequential testing methods built for that purpose, not implemented here. This simulation also collapses the tenure-driven heterogeneous effect from section 2 into two flat rates; a contextual bandit that used tenure the way the CATE model does is the natural next step.

## 2. Uplift/CATE modeling: who actually benefits

The +4.2pp average lift is real, but it's an average across everyone, and averages can hide that some users benefit far more than others. A T-learner (two gradient-boosted classifiers, one fit per arm, on tenure, recent session count, and pre-period revenue) predicts each held-out test user's individual treatment effect, validated the standard way for uplift models: by checking whether a higher predicted effect actually corresponds to a bigger realized effect on data the model never saw during fitting.

| | |
|---|---|
| Realized lift, top predicted-CATE quintile vs. bottom | +8.0pp vs. +1.0pp |
| Qini coefficient (targeting by predicted CATE vs. random) | 52.1 |
| Predicted CATE, newest users (0-33 days) vs. longest-tenured (320+ days) | +11.8pp vs. -1.3pp |

Predicted effect tracks realized effect on held-out data (Figure 3), and the model recovers platform tenure as the driver of the heterogeneity without being told to look for it (Figure 4): newer users, who haven't yet learned the old reminder flow, get most of the benefit from a clearer one; long-tenured users see essentially none.

![Uplift calibration](reports/figures/uplift_calibration.png)

*Figure 3. Predicted CATE vs. realized lift, by quintile of predicted effect.*

![Predicted CATE by tenure](reports/figures/uplift_by_tenure.png)

*Figure 4. Predicted CATE by platform-tenure bucket.*

### A second, more rigorous CATE estimator

The T-learner's weakness is structural: differencing two independently-fit models amplifies whatever noise each one picked up on its own. [EconML](https://github.com/py-why/econml)'s `CausalForestDML` avoids that by orthogonalizing the outcome and treatment against the covariates first (the "double" in double machine learning), then fitting a causal forest on what's left. Trained on the identical split, covariates, and held-out test set as the T-learner above, it nearly doubles the Qini coefficient:

| | |
|---|---|
| Qini coefficient, T-learner vs. CausalForestDML | 52.1 vs. 95.1 |
| CausalForestDML average treatment effect | +4.7pp (95% CI: 1.1 to 8.3pp) |

![Qini comparison, T-learner vs. CausalForestDML](reports/figures/cate_econml_qini_comparison.png)

*Figure 5. Qini curves, T-learner vs. CausalForestDML, on the identical held-out test set.*

The T-learner's Qini curve still clears random targeting by a wide margin, so it was never a bad model (Figure 5). The gap is what a hand-rolled two-model difference costs against an estimator built specifically to avoid the noise-amplification problem that causes.

## 3. Difference-in-differences: regional rollout

A new in-app collections feature was rolled out to 20 of 40 regions first, based on business priority rather than random assignment, which rules out a simple before/after read. A region and day fixed-effects regression, checked against a pre-period parallel-trends test first, isolates the treatment effect from any shared time trend.

| | |
|---|---|
| Pre-period trend difference (placebo check) | Not significant (p = 0.70), supports parallel trends |
| DiD estimate | +4.0pp on-time repayment (95% CI: 3.7 to 4.3pp), p < 0.0001 |

Treated and control regions move together before rollout and diverge after (Figure 6).

![Parallel trends](reports/figures/did_parallel_trends.png)

*Figure 6. On-time repayment rate by day, treated vs. control regions.*

### Three robustness checks on the DiD estimate

Standard errors are clustered by region, but 40 regions with 20 treated is on the edge of "few clusters" territory where the usual asymptotic cluster-robust standard errors can be unreliable. A wild cluster bootstrap (Cameron, Gelbach & Miller 2008) recomputes inference on the same coefficient without relying on that asymptotic approximation, using the [diff-diff](https://github.com/igerber/diff-diff) package: 95% CI of 3.7 to 4.3pp, matching the analytical result almost exactly, which is itself evidence the cluster count here isn't causing problems.

The pre-trends placebo check above fails to detect a trend difference, which is a weaker statement than proving none exists. Honest DiD (Rambachan & Roth 2023) quantifies the gap: how large would an undetected post-period violation of parallel trends have to be, relative to the largest pre-period wobble actually observed, before the effect stops being distinguishable from zero. Here that breakdown point is M = 0.35, meaning the conclusion holds only if any unmeasured drift after rollout stays under about a third of the size of the noisiest pre-period swing (Figure 7).

![Honest DiD sensitivity](reports/figures/did_honest_sensitivity.png)

*Figure 7. Honest DiD sensitivity analysis: breakdown point M.*

A third check comes from [DoWhy](https://github.com/py-why/dowhy)'s model/identify/estimate/refute framework, which formalizes the causal graph explicitly (region and day as the adjustment set) and runs a generic refutation suite orthogonal to the DiD-specific checks above: adding a random confounder, replacing the real treatment with a permuted placebo, and refitting on random 80% subsets of the data. The estimate survives all three (Figure 8): it barely moves when a random confounder is added or the data is subsetted, and it collapses to essentially zero under a placebo treatment, exactly the pattern that says the effect is real rather than an artifact of the estimation procedure.

![DoWhy refutation suite](reports/figures/dowhy_refutation.png)

*Figure 8. DoWhy refutation suite: original estimate vs. each refuter's re-estimated effect.*

## 4. RFM customer segmentation

Recency, frequency, and monetary value, clustered with KMeans (k chosen by silhouette score, not fixed in advance) into three segments.

| Segment | % of customers | % of revenue |
|---|---|---|
| Champions | 16% | 46% |
| Loyal | 52% | 48% |
| Dormant | 32% | 6% |

Champions are 16% of customers and 46% of revenue; Dormant customers are nearly a third of the base and 6% of revenue (Figure 9).

![Segment revenue share](reports/figures/segment_revenue_share.png)

*Figure 9. Share of customers and share of revenue, by RFM segment.*

## 5. Support ticket topic modeling

TF-IDF + NMF on 1,500 synthetic support tickets recovers five topics from text alone, validated at 85% purity against the known ground-truth categories (a check only possible because the data is synthetic) (Figure 10).

![Topic volume](reports/figures/topic_volume.png)

*Figure 10. Ticket volume by recovered topic.*

One topic, general account questions, doesn't cluster cleanly on its own; it scatters across the other four because it shares vocabulary with them rather than having a distinct signature. That's a real limitation of unsupervised topic modeling on short text.

## Recommendation

Ship the reminder redesign; the lift is well outside noise and confirmed two ways (a standard test and a CUPED-adjusted one with a tighter interval). For future tests in this same low-stakes, high-traffic category, weigh the fixed-horizon design's clean inference against a Thompson Sampling design's lower regret: the simulation above suggests roughly 900 conversions over 40,000 users, real revenue, were the price paid here for a fixed design's valid p-value. But ship it targeted, not blanket: the uplift model shows the benefit concentrates heavily in newer users, so rolling the redesign out to long-tenured users buys almost nothing while the engineering and support cost of maintaining two reminder flows is the same either way. For the regional rollout, the fixed-effects estimate, the pre-period placebo check, and both robustness checks (wild cluster bootstrap, Honest DiD) support treating the +4.0pp effect as real rather than a pre-existing regional difference, which makes the case for extending the rollout to the remaining regions. For lifecycle marketing, the RFM segments show where a differentiated offer would pay off most: roughly a third of customers are Dormant and contribute barely 6% of revenue, so a win-back offer targeted at that group would cost little in forgone Champion/Loyal attention. For support operations, route the four cleanly-separated topics to a keyword/topic-based triage rule, but keep a human or supervised classifier in the loop for general account questions, since that category doesn't have a clean unsupervised signature to route on.

## Repo layout

- `notebooks/03_growth_experimentation_segmentation.ipynb`: full technical walkthrough, executed with all charts and results inline.
- `src/`: the reproducible pipeline (data generation, experiment design/CUPED, sequential experimentation/bandits, uplift/CATE modeling and its EconML comparison, causal inference and its robustness checks including the DoWhy refutation suite, segmentation, topic modeling) as standalone scripts.
- `tests/`: pytest suite covering data-generation invariants, the fixed-horizon vs. Thompson Sampling simulation, the DiD estimator and its robustness checks (against synthetic panels with a known injected effect or a known injected pre-trend violation), the DoWhy refutation suite, the uplift model's bucket-calibration and Qini-curve logic, the CausalForestDML comparison, and the RFM/topic-modeling helper functions.
- `reports/`: generated charts and CSV outputs.

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/experiment_design.py
python src/sequential_experimentation.py
python src/uplift_modeling.py
python src/cate_econml.py
python src/causal_inference.py
python src/robustness_checks.py
python src/dowhy_refutation.py
python src/segmentation.py
python src/ticket_topics.py
```

`data/` and `reports/*.csv` are gitignored; regenerate them by running the scripts above.

## Tests

```bash
pytest tests/ -v
```

Runs in CI on every push (see the badge at the [repo root](../../README.md)).
