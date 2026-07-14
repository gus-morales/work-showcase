-- Raw daily ops panel, one row per day. Detectors (src/detectors.py)
-- compute their own rolling windows, so this stays unaggregated.
select
    day,
    date,
    pipeline_duration_minutes,
    pipeline_success_rate,
    data_freshness_hours,
    row_count
from pipeline_runs
order by day;
