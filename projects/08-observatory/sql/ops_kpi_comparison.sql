-- Latest day's ops KPIs vs. the same metrics 7 days earlier. An ad hoc
-- query an analyst would run directly against the database, not part
-- of the automated snapshot pipeline.
with latest as (
    select * from pipeline_runs order by day desc limit 1
),
week_ago as (
    select pipeline_runs.*
    from pipeline_runs, latest
    where pipeline_runs.day = latest.day - 7
)
select
    latest.day as latest_day,
    latest.pipeline_duration_minutes as duration_now,
    week_ago.pipeline_duration_minutes as duration_7d_ago,
    latest.pipeline_success_rate as success_rate_now,
    week_ago.pipeline_success_rate as success_rate_7d_ago,
    latest.data_freshness_hours as freshness_now,
    week_ago.data_freshness_hours as freshness_7d_ago
from latest
cross join week_ago;
