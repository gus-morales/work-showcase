-- Channel mix by acquisition cohort, using a window function to get each
-- channel's share of that month's acquisitions. This is what feeds the
-- "growth masking a mix shift toward lower-quality channels" finding.
SELECT
    cohort_month,
    acquisition_channel,
    COUNT(*) AS customers,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY cohort_month), 3) AS channel_share
FROM customers
GROUP BY 1, 2
ORDER BY 1, 2;
