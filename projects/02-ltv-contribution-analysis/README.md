# Customer LTV & Contribution Analysis

A growth analysis for a synthetic buy now, pay later (BNPL) fintech: what's driving GMV growth, which acquisition channels actually produce valuable customers, and how to predict a customer's lifetime value from their first 30 days. Built on synthetic transaction data, mirroring the growth/contribution analysis work I did alongside credit risk modeling in fintech.

**For the full technical walkthrough (SQL, BG/NBD + Gamma-Gamma CLV modeling, calibration/holdout validation, gradient boosting), see the [notebook](notebooks/02_ltv_contribution_analysis.ipynb).** This README is the short version.

> All data here is synthetically generated. No proprietary data, models, or results from any employer are used or implied. This is the same fictional company as project 01, viewed from the growth/LTV side instead of the credit-risk side.

**Skills and tools featured:**

- SQL (ranking and offset window functions, moving averages, CTEs, cohort joins via DuckDB)
- Data contracts and a governed metric layer
- Cohort retention analysis
- Log-share contribution decomposition
- Probabilistic customer lifetime value (BG/NBD + Gamma-Gamma) with calibration/holdout validation
- Gradient boosting regression with feature importance

## The problem

GMV was up sharply over two years, which looks like a growth story on the surface. The question is whether that growth is healthy, more valuable customers coming back and spending more, or whether it's being propped up by pouring more people into the top of the funnel while the customer base underneath gets less engaged.

## What this does

GMV is just active customers times orders per customer times average order value, multiplied together. This project decomposes GMV growth into those three drivers using SQL-aggregated monthly KPIs, so it's clear which one is actually responsible for the trend, then checks whether the answer traces back to a shift in acquisition channel mix.

Separately, it builds two complementary customer lifetime value models: a probabilistic model that scores customers from their full transaction history, and a machine learning model that scores customers from just their first 30 days, for cases where a full history isn't available yet.

## Data contracts and a governed metric layer

Before anything downstream trusts `customers.csv` or `orders.csv`, `src/contracts.py` checks their schema, category values, value ranges, and that the two tables reference each other correctly. This runs automatically, so a broken or stale file fails loudly at the first script that touches it, rather than quietly producing a wrong chart three steps later.

`src/metrics.py` centralizes the definitions this project actually uses: GMV is total order value, revenue is total fee revenue, and the two are not the same number, revenue is GMV times the take rate. Adding this layer surfaced a real inconsistency: the early-life CLV model was labeling its GMV target "revenue." The underlying numbers were never wrong, since GMV and revenue move in lockstep here at a constant take rate, so nothing downstream was affected. But the labels are now corrected so the two metrics can't silently drift apart as this project grows.

## Results

GMV grew from $868K to $3.36M between month 4 and month 20. Decomposing that change shows active customer growth alone accounts for essentially all of it. Order frequency and average order value were each a net drag over the same period.

| | |
|---|---|
| GMV growth, month 4 to month 20 | +$2.50M |
| Share of growth from new customers | ~107% (frequency and order value are net drags) |
| Best vs. worst channel, revenue per customer | Partner store $322 vs. paid social $120 (2.7x) |
| Paid social's share of new cohorts | Roughly tripled (16% to 47%) |
| Predicted 12-month CLV captured by top decile | 52.7% (the top 10% of customers by predicted value account for this share of actual realized value) |
| Early-life model (day-30 features), holdout R² | 0.37 (the model explains 37% of the variance in 12-month GMV on data it wasn't trained on) |

Active customer growth alone accounts for essentially all of the GMV change (Figure 1).

![GMV contribution by driver](reports/figures/contribution_monthly.png)

*Figure 1. GMV growth decomposed into active customers, orders per customer, and average order value.*

The channel data explains why growth looks healthier than it is: paid social is both the fastest-growing acquisition channel and the lowest-quality one by a wide margin (Figure 2), and its share of new customer cohorts has roughly tripled (Figure 3).

![Channel quality](reports/figures/channel_quality.png)

*Figure 2. Revenue per customer by acquisition channel.*

![Channel mix shift](reports/figures/channel_mix_shift.png)

*Figure 3. Acquisition channel mix by cohort month.*

## Two ways to estimate customer value

The first model, BG/NBD + Gamma-Gamma, is the standard probabilistic pairing for this problem. One part predicts how many more purchases a customer will make and the odds they've already churned, based purely on how often and how recently they've bought before. The other part adds a dollar estimate on top, based on their typical order size. Together they need no engineered features, just transaction history, but that history has to build up over months before the estimate stabilizes.

It was validated against a 6-month calibration/holdout split: fit on an earlier window, then checked against what customers actually did in the following one. Its predicted purchase counts track actual holdout behavior closely (Figure 4).

![CLV calibration holdout](reports/figures/clv_calibration_holdout.png)

*Figure 4. Predicted vs. actual purchase counts, calibration/holdout validation.*

The second model fills the gap for a brand-new customer: a gradient boosting regressor trained on just their first 30 days of behavior, which explains about 37% of the variance in their eventual 12-month GMV (Figure 5). That's a real signal, order count and spend in the first month dominate the prediction, but it's a partial one. Some customers front-load a single purchase and then churn, and 30 days of data alone can't fully tell that apart from a customer who's just getting started.

![Early life predicted vs actual](reports/figures/early_life_predicted_vs_actual.png)

*Figure 5. Early-life model: predicted vs. actual 12-month GMV.*

## Growth trend and revenue concentration

The decomposition above explains what's driving GMV between two snapshots (month 4 vs. month 20). Two more questions need the full monthly series rather than those two points alone: is the growth *rate* itself holding up, and how much of any given cohort's revenue rides on a handful of its biggest spenders.

For the first, a SQL window function does the work directly: `LAG()` pulls each month's prior-month GMV to compute month-over-month growth, and a moving-average window frame (`AVG() OVER (... ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`) smooths that noisy month-to-month number into a clearer trend line. GMV is still growing every month, but the pace is not: average MoM growth runs 28.6% across months 2-6 and has fallen to 5.3% across months 20-24 (Figure 6). That's a materially different growth story than "GMV tripled," even though both statements are true at once.

![GMV trend with moving average](reports/figures/gmv_trend_moving_average.png)

*Figure 6. Monthly GMV vs. a trailing 3-month moving average.*

For the second, `DENSE_RANK()` partitioned by acquisition cohort ranks customers by revenue within their own cohort, normalized to each cohort's top 5% rather than a fixed headcount, since cohorts range from roughly 300 to 950 customers and a fixed count of 5 would look artificially more "concentrated" in a small cohort than a large one. Across all 24 cohorts, that top 5% consistently accounts for about a quarter of the cohort's total revenue (26.0% on average, ranging from 20.3% to 35.5%), a fairly stable Pareto-style pattern rather than something that drifts as the platform scales (Figure 7).

![Revenue concentration by cohort](reports/figures/revenue_concentration_by_cohort.png)

*Figure 7. Share of cohort revenue from its top 5% of customers by spend, by acquisition cohort.*

## Recommendation

The GMV trend is not the health signal it looks like. Growth is increasingly funded by paid social, a channel that produces customers worth roughly a third as much as the best channel, while order frequency and order value are both quietly declining underneath the growth line, and the MoM growth rate itself has slowed from the high-20s to the mid-single digits. Before scaling paid social spend further, the marginal customer acquisition cost (CAC) on that channel should be checked against its ~$120 average lifetime revenue, not against blended GMV growth.

The steady ~26% top-5% revenue concentration within each cohort is worth watching but not currently a red flag: it's stable rather than worsening, so growth isn't becoming more dependent on a shrinking set of whales over time. A VIP retention program would still be worth the investment given the concentration is real, just not because it's getting worse.

For lifetime value scoring in production, use the day-30 model to triage new customers into engagement tiers right after signup, then let the BG/NBD + Gamma-Gamma model take over once a customer has enough transaction history for it to stabilize, rather than picking one model for the whole customer lifecycle.

## Repo layout

- `notebooks/02_ltv_contribution_analysis.ipynb`: full technical walkthrough, executed with all charts and results inline.
- `sql/`: cohort revenue, monthly KPIs, channel quality, channel mix-shift, month-over-month growth with a moving average, and cohort revenue-concentration queries, run via DuckDB (an embedded analytical SQL engine that queries the CSVs directly, no server to stand up).
- `src/`: the reproducible pipeline (data generation, data contracts, the metric registry, SQL runner, contribution decomposition, channel analysis, growth trend, revenue concentration, CLV modeling) as standalone scripts.
- `tests/`: pytest suite covering data-generation invariants, the data contracts, the metric registry's SQL-consistency check (against this project's actual sql/*.sql files), the SQL queries (against a temp dataset), and the log-share decomposition arithmetic.
- `reports/`: generated charts and CSV outputs.

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/cohort_analysis.py
python src/contribution.py
python src/channel_analysis.py
python src/growth_trend.py
python src/revenue_concentration.py
python src/clv_model.py
```

`data/` and `reports/` are gitignored except for what's needed to render this README; regenerate them by running the scripts above.

## Tests

```bash
pytest tests/ -v
```

Runs in CI on every push (see the badge at the [repo root](../../README.md)).
