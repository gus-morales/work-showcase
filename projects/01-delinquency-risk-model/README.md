# BNPL Delinquency Risk Model

A risk model that flags which buy-now-pay-later loans are likely to go 30+ days past due. It picks its approve/decline threshold from actual business costs instead of a default 0.5 cutoff, checks that its decisions don't disadvantage any group unfairly, looks at how fast different segments default rather than just whether they do, and ships as a small scoring service to prove the whole thing actually runs. Built on synthetic data modeled after a BNPL lending book, mirroring delinquency-prediction work I did in fintech.

**For the full technical walkthrough (modeling, calibration, SHAP, drift monitoring), see the [notebook](notebooks/01_delinquency_risk_model.ipynb).** This README is the short version. For a validation-style write-up (data lineage, conceptual soundness, outcomes analysis, ongoing monitoring plan), see the [model validation memo](docs/model_validation_memo.md).

> All data here is synthetically generated. No proprietary data, models, or results from any employer are used or implied.

**Skills and tools featured:**

- Classification modeling (logistic regression + gradient-boosted trees)
- Leakage-safe feature pipelines (feature-engine, fit on the training split only)
- Time-series cross-validated hyperparameter search
- Probability calibration
- Cost-based decision optimization
- SHAP interpretability
- Drift and calibration monitoring
- Fair lending analysis (disparate impact / four-fifths rule, SHAP-derived adverse action reason codes)
- Survival analysis (Kaplan-Meier, Cox proportional hazards)
- Model serving (FastAPI scoring endpoint over the trained pipeline)

## The problem

A BNPL lender approves a loan in seconds at checkout. Approve a customer who pays on time, and the lender earns a small merchant fee. Approve one who defaults, and the lender loses most of the loan amount. Getting the approve/decline line right is worth real money.

## What this does

Trains a model to predict delinquency risk at loan approval, then picks the approve/decline threshold that minimizes expected losses given realistic cost assumptions, rather than using a generic 0.5 cutoff.

## Results

Two models were compared: a logistic regression baseline, and a gradient-boosted trees model with hyperparameters tuned by cross-validation. They came out close on AUC (a 0-to-1 score for how well a model ranks risky loans above safe ones): 0.79 for gradient boosting, 0.80 for logistic regression. That closeness says the underlying patterns here are fairly straightforward, not deeply nonlinear. Gradient boosting was kept anyway, since it handles interactions between features better as more features get added, and it's what the SHAP interpretability step further down relies on.

Before those probabilities can be used for a real decision, they need one more step: calibration. Gradient-boosted models tend to output probabilities that are more extreme than they should be, too close to 0 or 1. Isotonic calibration corrects that, so a predicted "10% chance of default" actually means close to 10% in practice.

With calibrated probabilities in hand, the next question is where to draw the approve/decline line. A default 0.5 cutoff ignores that the two kinds of mistakes cost very different amounts: missing a loan that goes delinquent costs roughly 70% of its principal, while wrongly declining a loan that would have been repaid only costs a 6% fee. A missed default is about 12x more expensive than a wrongly declined loan, so the model should lean toward declining. Testing every possible cutoff against expected cost confirms exactly how far: the optimal threshold lands at 0.05, and using it instead of 0.5 cuts expected losses by **67%**.

| | |
|---|---|
| Model accuracy (AUC), held-out test | 0.79 |
| Model accuracy (AUC), cross-validated | 0.81 ± 0.005 |
| Expected loss reduction vs. a naive 0.5 cutoff | 67% |
| Share of actual delinquent loans caught | 92% |

Delinquency spikes in the last three months of data, which carry a simulated economic shock (Figure 1). Testing every possible threshold against expected cost shows exactly where that 67% improvement comes from (Figure 2).

![Delinquency by month](reports/figures/delinquency_by_month.png)

*Figure 1. Delinquency rate by loan origination month. The last three months carry a simulated macro shock.*

![Threshold selection](reports/figures/threshold_cost_curve.png)

*Figure 2. Expected portfolio cost by decision threshold, cost-optimal threshold vs. the naive 0.5 cutoff.*

## Missing bureau scores

About 9% of applicants have no credit bureau score on file: gig workers, informal workers, and people who just joined the platform. That's routine for a BNPL lender, not an edge case, so the model has to handle it directly instead of dropping those applicants.

The fix has two parts. First, fill in the missing score with the median score across all applicants, a safe default that isn't thrown off by outliers. Second, add a separate yes/no flag marking that the score was missing in the first place, since that fact alone can be informative, on top of whatever value gets filled in.

Bureau score turns out to be the single strongest signal behind the model's predictions, measured by SHAP (a standard way to see how much each feature pushed a given prediction up or down). The missing-score flag also carries a small amount of signal on its own, less than bureau score itself, but enough that "no credit history" tells the model something employment type and tenure don't already cover.

## Calibration drift under a simulated shock

To see how the model holds up when conditions change, it was stress-tested against a simulated economic shock (higher rates, tighter budgets) covering the last three months of data.

The usual way to catch this kind of problem is input-drift monitoring: checking whether the feature distributions in the current data still look like the reference period they were trained on. The standard metric for that is PSI (Population Stability Index), where a value above roughly 0.2 signals a meaningful shift. Every feature's PSI stayed well under that line, and even the rate of missing bureau scores barely moved (9.4% reference vs. 9.9% monitored). By that measure, nothing looked wrong.

But the actual delinquency rate rose anyway, and the model quietly started under-predicting risk. Input-drift monitoring missed this because the inputs themselves hadn't shifted, the relationship between the inputs and the outcome had. Catching it required a second, different check: comparing the model's predicted delinquency rate against the actual one, month by month (Figure 3).

![Monitoring](reports/figures/drift_predicted_vs_actual.png)

*Figure 3. Predicted vs. actual delinquency rate by month, reference window vs. the monitored (shock) window.*

### Rate-mix shift decomposition

One more explanation needed ruling out: maybe the portfolio had quietly shifted toward riskier customers, more gig workers, say, which would raise the overall delinquency rate without tripping any single feature's PSI.

A rate-mix decomposition checks this by splitting the change in the overall delinquency rate into two pieces: how much came from the customer mix shifting, and how much came from existing segments simply getting riskier on their own. The result: 96% of the increase is the second kind, segments getting riskier, not a change in who's being approved (Figure 4). Every employment type moved in the same direction together, which rules out one bad segment dragging up the average by itself (Figure 5).

| | |
|---|---|
| Delinquency rate, reference vs. monitored window | 13.5% vs. 19.1% |
| Share of the increase from composition shift (mix effect) | 3.6% |
| Share of the increase from within-segment rate change | 96.4% |

![Rate-mix shift decomposition](reports/figures/rate_mix_shift_decomposition.png)

*Figure 4. Rate-mix shift decomposition: mix effect (composition shift) vs. rate effect (within-segment change).*

![Rate-mix shift by segment](reports/figures/rate_mix_shift_by_segment.png)

*Figure 5. Delinquency rate by employment-type segment, reference vs. monitored window.*

## Fair lending review

A lending model doesn't need to use race, gender, or other protected characteristics directly to end up treating those groups unequally. If other data the model does use happens to correlate with one of those characteristics, unequal outcomes can show up without the model ever being told what the characteristic is. This section checks whether that happened here.

Since this project runs on synthetic data, there's no real demographic data to test against. A stand-in column, `demographic_group`, is added instead: generated with a mild correlation to city tier, but never shown to the model. It exists only so the model's approve/decline decisions can be checked against it afterward.

The standard test for this is the four-fifths rule: a group's approval rate should be at least 80% of the highest-approving group's rate. This model clears that comfortably, and a two-proportion z-test confirms the small gap between groups isn't statistically distinguishable from noise (Figure 6).

| | |
|---|---|
| Approval rate, Group A vs. Group B (reference) | 37.0% vs. 38.5% |
| Disparate impact ratio | 0.961 (passes the 0.80 four-fifths threshold) |
| Statistical significance of the gap | Not significant, p = 0.46 |

![Fair lending approval rate](reports/figures/fair_lending_approval_rate.png)

*Figure 6. Approval rate by demographic group vs. the four-fifths rule.*

Every declined applicant also gets an adverse action reason code, the specific reason a lender is legally required to give a rejected applicant under the Equal Credit Opportunity Act (ECOA). Those reasons come from SHAP but are restricted to a fixed list of legitimate credit factors: things like city, device type, or acquisition channel never appear as a reason, even when they show up in the SHAP breakdown, since a lender shouldn't cite an applicant's neighborhood or how they signed up as grounds for a decline.

![Fair lending reason codes](reports/figures/fair_lending_reason_codes.png)

*Figure 7. Primary adverse action reason among declined applicants.*

## Time-to-default

The model above treats delinquency as a single yes/no outcome inside a fixed window. But a loan that goes bad in week 2 is a very different problem than one that goes bad in month 10, even though both get labeled "delinquent" the same way. Survival analysis keeps that time dimension instead of collapsing it away.

A Kaplan-Meier curve tracks, for each employment type, what share of loans are still current as time passes. By the end of the observation window: 93.1% of salaried loans are still current, against 82.7% self-employed, 72.4% gig-economy, and 66.8% informal (Figure 8). That gap is far too large to be random chance, a log-rank test (the standard significance test for comparing survival curves across groups) puts it at p < 0.001.

A Cox proportional hazards model turns that same gap into one number per group: how much faster a group defaults relative to salaried loans, everything else held equal. Informal-employment loans default at 2.18x the salaried rate, gig-economy at 1.92x, self-employed at 1.42x (Figure 9). Down payment ratio works the other way, the strongest protective factor in the model, in the same direction it has on the classification model above.

| | |
|---|---|
| Concordance index (Cox model) | 0.814 |
| Model accuracy (AUC), classification model, held-out test | 0.79 |
| Log-rank test across employment types | p < 0.001 |

The concordance index (0.81) is the survival model's version of AUC: the probability that, given two random loans, it ranks the one that defaults first as the riskier one. It lands close to the classification model's own AUC (0.79), so both models agree on how separable the underlying risk actually is. The classification model still makes the real-time approve/decline call, that's what it's built for. The survival model's job starts after that: once a loan is already on the books, it tells collections which segments are worth chasing first.

![Time-to-default by employment type](reports/figures/survival_km_by_employment.png)

*Figure 8. Kaplan-Meier estimate of time-to-default by employment type.*

![Cox hazard ratios](reports/figures/survival_hazard_ratios.png)

*Figure 9. Cox proportional hazards ratios for time-to-default, with 95% confidence intervals.*

## Serving the model

Everything above only proves the model works offline, against a saved dataset. `src/serve.py` proves it also works as a live service: it wraps the trained pipeline in a small FastAPI app with one endpoint, `POST /predict`.

Send it an applicant's raw data as JSON, and it runs the same feature engineering used at training time, then returns a probability plus an approve/decline call. `credit_bureau_score` is an optional field, matching how the model handles missing scores everywhere else in this project: about 9% of real applicants would show up thin-file. Pydantic (a library that validates incoming request data against a declared schema) rejects malformed input, out-of-range values, wrong categories, missing fields, before it ever reaches the model.

```bash
uvicorn serve:app --reload   # from src/
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 34, "monthly_income_usd": 2400, "tenure_months_platform": 8,
  "num_previous_loans": 3, "credit_bureau_score": 690,
  "avg_prior_repayment_delay_days": 1.5, "num_active_loans_elsewhere": 1,
  "num_installments": 6, "loan_amount_usd": 850, "down_payment_ratio": 0.15,
  "city_tier": "tier1", "employment_type": "salaried", "device_type": "ios",
  "acquisition_channel": "organic", "merchant_category": "electronics"
}'
# {"delinquency_probability": 0.0759, "decision": "decline", "threshold": 0.05}
```

This is deliberately small: no batching, auth, model versioning, or canary rollout, none of which this project is trying to demonstrate. What it does prove is that the training artifacts round-trip cleanly into something that scores a single application the same way the offline evaluation did.

## Recommendation

Ship the cost-based threshold over the naive 0.5 cutoff, that's where the 67% expected-loss reduction comes from. But ship it with calibration-gap monitoring running alongside standard PSI checks, not in place of it: this model would have looked healthy on every input-drift dashboard while quietly under-pricing risk through the shock, the kind of gap that only shows up in a loss report a quarter later if nobody's watching for it directly. Since the rate-mix decomposition rules out a change in who's being approved, the fix belongs in the model itself, retrain or recalibrate on shock-period data, rather than in underwriting policy toward any particular segment.

The fair lending review currently passes with a comfortable margin, but it's worth re-running on the same cadence as the drift checks above, not treated as a one-time clearance. Downstream of underwriting, the survival model's segment-level speed-to-default is a reasonable input to how collections prioritizes outreach: informal and gig-economy loans default both more often and faster. And since the model is already proven servable, none of this has to wait on a separate deployment project to act on.

## Repo layout

- `notebooks/01_delinquency_risk_model.ipynb`: full technical walkthrough, executed with all charts and results inline.
- `src/`: the reproducible pipeline (data generation, features, training, interpretability, monitoring, rate-mix shift decomposition, fair lending review, survival analysis, model serving) as standalone scripts.
- `tests/`: pytest suite covering data-generation invariants, the feature-engineering functions and pipeline (including the missing-bureau-score handling), the rate-mix shift decomposition, the fair lending disparate-impact and reason-code logic, the survival-analysis covariate construction and Cox fit, and the serving endpoint's request validation and scoring behavior.
- `reports/`: generated charts, metrics, and monitoring reports.
- `docs/model_validation_memo.md`: SR 11-7-style validation write-up (data lineage, conceptual soundness, outcomes analysis, ongoing monitoring plan).

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/eda.py
python src/tune.py       # hyperparameter search, writes reports/best_params.json
python src/train.py      # picks up best_params.json automatically if present
python src/interpret.py
python src/monitor_drift.py
python src/rate_mix_shift.py
python src/fair_lending.py
python src/survival_analysis.py
```

`data/` and `reports/model.pkl` are gitignored; regenerate them by running the scripts above. `reports/best_params.json` is committed so `train.py` reproduces the same tuned model without re-running the search.

## Tests

```bash
pytest tests/ -v
```

Runs in CI on every push (see the badge at the [repo root](../../README.md)).
