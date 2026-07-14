-- Daily scored-customer volume with a 7-day rolling average, computed
-- in SQL via a window function. An ad hoc query, not part of the
-- automated snapshot pipeline.
with daily_volume as (
    select day, date, count(*) as customers_scored
    from scoring_log
    group by day, date
)
select
    day,
    date,
    customers_scored,
    avg(customers_scored) over (
        order by day
        rows between 6 preceding and current row
    ) as rolling_7d_avg
from daily_volume
order by day;
