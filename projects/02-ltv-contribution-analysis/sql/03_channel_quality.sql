-- Acquisition channel quality: average orders and revenue per customer by
-- channel, over the customer's full observed lifetime so far.
WITH channel_orders AS (
    SELECT
        c.acquisition_channel,
        o.customer_id,
        COUNT(*) AS n_orders,
        SUM(o.fee_revenue_usd) AS revenue_usd
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    GROUP BY 1, 2
)
SELECT
    acquisition_channel,
    COUNT(*) AS customers,
    ROUND(AVG(n_orders), 2) AS avg_orders_per_customer,
    ROUND(AVG(revenue_usd), 2) AS avg_revenue_per_customer_usd,
    ROUND(SUM(revenue_usd), 2) AS total_revenue_usd
FROM channel_orders
GROUP BY 1
ORDER BY avg_revenue_per_customer_usd DESC;
