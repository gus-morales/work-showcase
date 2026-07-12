-- Within each acquisition cohort, rank customers by total revenue and
-- flag the top 5% (not a fixed headcount, cohorts range from ~300 to
-- ~950 customers, so a fixed count of 5 would mechanically look more
-- "concentrated" for small cohorts and less for large ones regardless
-- of actual concentration). DENSE_RANK partitioned by cohort_month,
-- normalized against each cohort's own size.
WITH customer_revenue AS (
    SELECT
        c.cohort_month,
        c.customer_id,
        SUM(o.fee_revenue_usd) AS revenue_usd
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    GROUP BY 1, 2
),
ranked AS (
    SELECT
        cohort_month,
        customer_id,
        revenue_usd,
        DENSE_RANK() OVER (PARTITION BY cohort_month ORDER BY revenue_usd DESC) AS revenue_rank,
        COUNT(*) OVER (PARTITION BY cohort_month) AS cohort_size
    FROM customer_revenue
)
SELECT
    cohort_month,
    MAX(cohort_size) AS cohort_size,
    ROUND(SUM(CASE WHEN revenue_rank <= GREATEST(1, CAST(cohort_size * 0.05 AS INTEGER))
              THEN revenue_usd ELSE 0 END), 2) AS top5pct_revenue_usd,
    ROUND(SUM(revenue_usd), 2) AS cohort_total_revenue_usd,
    ROUND(
        SUM(CASE WHEN revenue_rank <= GREATEST(1, CAST(cohort_size * 0.05 AS INTEGER))
            THEN revenue_usd ELSE 0 END) * 1.0
        / NULLIF(SUM(revenue_usd), 0), 4
    ) AS top5pct_revenue_share
FROM ranked
GROUP BY 1
ORDER BY 1;
