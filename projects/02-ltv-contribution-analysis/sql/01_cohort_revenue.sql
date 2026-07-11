-- Monthly cohort revenue/retention curve: for each acquisition cohort,
-- how many of the originally acquired customers are still ordering N
-- months later, and how much revenue does the cohort produce per
-- originally-acquired customer (not per still-active customer, since
-- that measure quietly hides churn).
WITH cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM customers
    GROUP BY cohort_month
),
cohort_activity AS (
    SELECT
        c.cohort_month,
        o.months_since_acquisition,
        COUNT(DISTINCT o.customer_id) AS active_customers,
        SUM(o.fee_revenue_usd) AS revenue_usd
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    GROUP BY 1, 2
)
SELECT
    a.cohort_month,
    a.months_since_acquisition,
    a.active_customers,
    s.cohort_size,
    ROUND(a.active_customers * 1.0 / s.cohort_size, 4) AS retention_rate,
    ROUND(a.revenue_usd, 2) AS revenue_usd,
    ROUND(a.revenue_usd / s.cohort_size, 2) AS revenue_per_acquired_customer
FROM cohort_activity a
JOIN cohort_sizes s ON s.cohort_month = a.cohort_month
ORDER BY a.cohort_month, a.months_since_acquisition;
