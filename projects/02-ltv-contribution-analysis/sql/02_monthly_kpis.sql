-- Calendar-month KPIs, aggregated to the granularity a contribution
-- decomposition needs: GMV = active_customers x orders_per_customer x
-- avg_order_value. The decomposition arithmetic itself happens in Python
-- (src/contribution.py); this query just produces the clean monthly table.
SELECT
    order_month_index AS month_index,
    COUNT(DISTINCT customer_id) AS active_customers,
    COUNT(*) AS orders,
    ROUND(SUM(order_value_usd), 2) AS gmv_usd,
    ROUND(SUM(fee_revenue_usd), 2) AS revenue_usd,
    ROUND(SUM(order_value_usd) / COUNT(*), 2) AS avg_order_value_usd,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT customer_id), 3) AS orders_per_customer
FROM orders
GROUP BY 1
ORDER BY 1;
