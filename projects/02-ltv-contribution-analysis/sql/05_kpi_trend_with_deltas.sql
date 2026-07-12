-- Monthly GMV with month-over-month growth and a trailing 3-month
-- moving average, computed with LAG() and a moving window frame. The
-- moving average exists because MoM growth is noisy month to month;
-- smoothing it out is what actually shows whether growth is
-- decelerating, rather than only whether GMV is still rising.
WITH monthly AS (
    SELECT
        order_month_index AS month_index,
        SUM(order_value_usd) AS gmv_usd
    FROM orders
    GROUP BY 1
)
SELECT
    month_index,
    ROUND(gmv_usd, 2) AS gmv_usd,
    ROUND(LAG(gmv_usd) OVER (ORDER BY month_index), 2) AS prior_month_gmv_usd,
    ROUND(
        (gmv_usd - LAG(gmv_usd) OVER (ORDER BY month_index))
        / NULLIF(LAG(gmv_usd) OVER (ORDER BY month_index), 0) * 100, 2
    ) AS mom_growth_pct,
    ROUND(
        AVG(gmv_usd) OVER (ORDER BY month_index ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2
    ) AS gmv_3mo_moving_avg
FROM monthly
ORDER BY month_index;
