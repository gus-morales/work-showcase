-- Raw scoring log, one row per customer scored per day. popmon
-- (src/stability.py) does its own time-binning and histogramming, so
-- this feeds it unaggregated rows, not a pre-summarized panel.
select
    day,
    date,
    customer_id,
    tenure_months,
    monthly_usage_score,
    support_tickets_30d,
    plan_tier,
    predicted_churn_prob
from scoring_log
order by day, customer_id;
