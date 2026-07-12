# Customer LTV & Contribution Analysis

A growth analysis for a synthetic BNPL fintech: what's driving GMV growth, which acquisition channels actually produce valuable customers, and how to predict a customer's lifetime value from their first 30 days. Built on synthetic transaction data, mirroring the growth/contribution analysis work I do alongside credit risk modeling in fintech.

**For the full technical walkthrough (SQL, BG/NBD + Gamma-Gamma CLV modeling, calibration/holdout validation, gradient boosting), see the [notebook](notebooks/02_ltv_contribution_analysis.ipynb).** This README is the short version.

> All data here is synthetically generated. No proprietary data, models, or results from any employer are used or implied. This is the same fictional company as project 01, viewed from the growth/LTV side instead of the credit-risk side.

**Skills and tools featured:**

- SQL (window functions, CTEs, cohort joins via DuckDB)
- Data contracts and a governed metric layer
- Cohort retention analysis
- Log-share contribution decomposition
- Probabilistic customer lifetime value (BG/NBD + Gamma-Gamma) with calibration/holdout validation
- Gradient boosting regression with feature importance

## The problem

GMV was up sharply over two years, which looks like a growth story on the surface. The question is whether that growth is healthy, more valuable customers coming back and spending more, or whether it's being propped up by pouring more people into the top of the funnel while the customer base underneath gets less engaged.

## What this does

Decomposes GMV growth into its three multiplicative drivers (active customers, orders per customer, average order value) using SQL-aggregated monthly KPIs, then checks whether the answer traces back to a shift in acquisition channel mix. Separately, builds two complementary customer lifetime value models: a probabilistic BG/NBD + Gamma-Gamma model that scores customers from transaction history, and a gradient boosting model that scores customers from just their first 30 days.

## Data contracts and a governed metric layer

Before anything downstream trusts `customers.csv` or `orders.csv`, `src/contracts.py` checks schema, category values, value ranges, and referential integrity between the two tables (`db.get_connection()` runs this automatically, so a broken or stale file fails loudly at the first script that touches it rather than producing a quietly wrong chart three steps later).

`src/metrics.py` centralizes the definitions this project actually uses: GMV is `SUM(order_value_usd)`, revenue is `SUM(fee_revenue_usd)`, and they are not the same number (revenue is GMV times the take rate). Adding this layer surfaced a real inconsistency: the early-life CLV model was labeling its GMV target "revenue." The underlying numbers were never wrong, since GMV and revenue are proportional here with a constant take rate, so R² and the decile-capture read are unaffected. The labels are now corrected to GMV so the two metrics can't drift apart silently as this project grows.

## Results

GMV grew from $868K to $3.36M between month 4 and month 20. Decomposing that change shows active customer growth alone accounts for essentially all of it; order frequency and average order value are each net-negative contributors over the period.

| | |
|---|---|
| GMV growth, month 4 to month 20 | +$2.50M |
| Share of growth from new customers | ~107% (frequency and order value are net drags) |
| Best vs. worst channel, revenue per customer | Partner store $322 vs. paid social $120 (2.7x) |
| Paid social's share of new cohorts | Roughly tripled (16% to 47%) |
| Predicted 12-month CLV captured by top decile | 52.7% |
| Early-life model (day-30 features), holdout R² | 0.37 |

Active customer growth alone accounts for essentially all of the GMV change (Figure 1).

![GMV contribution by driver](reports/figures/contribution_monthly.png)

*Figure 1. GMV growth decomposed into active customers, orders per customer, and average order value.*

The channel data explains why: paid social is both the fastest-growing acquisition channel and the lowest-quality one by a wide margin (Figure 2), and its share of new cohorts has roughly tripled (Figure 3).

![Channel quality](reports/figures/channel_quality.png)

*Figure 2. Revenue per customer by acquisition channel.*

![Channel mix shift](reports/figures/channel_mix_shift.png)

*Figure 3. Acquisition channel mix by cohort month.*

## Two ways to estimate customer value

BG/NBD + Gamma-Gamma uses only transaction history and needs no features, but it takes months of purchase history to produce a stable read on a customer. Validated against a 6-month calibration/holdout split, its predicted purchase counts track actual holdout behavior closely (Figure 4).

![CLV calibration holdout](reports/figures/clv_calibration_holdout.png)

*Figure 4. Predicted vs. actual purchase counts, calibration/holdout validation.*

For a model that can score a customer immediately after signup, a gradient boosting regressor trained on day-30 behavior explains about 37% of the variance in 12-month GMV (Figure 5). That's a genuine signal, order count and spend in the first 30 days dominate the prediction, but it's a partial one: some customers front-load a purchase and then churn, which day-30 features alone can't fully separate from a customer who's just getting started.

![Early life predicted vs actual](reports/figures/early_life_predicted_vs_actual.png)

*Figure 5. Early-life model: predicted vs. actual 12-month GMV.*

## Recommendation

The GMV trend is not the health signal it looks like. Growth is increasingly funded by paid social, a channel that produces customers worth roughly a third as much as the best channel, and both order frequency and order value are quietly declining under the growth line. Before scaling paid social spend further, the marginal CAC on that channel should be checked against its ~$120 average lifetime revenue, not against blended GMV growth.

For lifetime value scoring in production, use the day-30 gradient boosting model to triage new customers into engagement tiers immediately after signup, then let the BG/NBD + Gamma-Gamma model take over once a customer has enough transaction history for it to stabilize, rather than picking one model for the whole customer lifecycle.

## Repo layout

- `notebooks/02_ltv_contribution_analysis.ipynb`: full technical walkthrough, executed with all charts and results inline.
- `sql/`: cohort revenue, monthly KPIs, channel quality, and channel mix-shift queries, run via DuckDB.
- `src/`: the reproducible pipeline (data generation, data contracts, the metric registry, SQL runner, contribution decomposition, channel analysis, CLV modeling) as standalone scripts.
- `tests/`: pytest suite covering data-generation invariants, the data contracts, the metric registry's SQL-consistency check (against this project's actual sql/*.sql files), the SQL queries (against a temp dataset), and the log-share decomposition arithmetic.
- `reports/`: generated charts and CSV outputs.

## Reproduce

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/cohort_analysis.py
python src/contribution.py
python src/channel_analysis.py
python src/clv_model.py
```

`data/` and `reports/` are gitignored except for what's needed to render this README; regenerate them by running the scripts above.

## Tests

```bash
pytest tests/ -v
```

Runs in CI on every push (see the badge at the [repo root](../../README.md)).
