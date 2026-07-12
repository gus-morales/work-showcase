# EDA summary

- Customers: 15,000, Orders: 94,358

- Orders per customer: mean 6.29, median 4, 90th percentile 14, 16.8% single-order customers

- Order value (USD): mean 581.52, median 486.34, 90th percentile 1075.78

- Missing values: 0 (data contracts in src/contracts.py enforce this at generation time)


## Acquisition channel mix

| acquisition_channel   | proportion   |
|:----------------------|:-------------|
| paid_social           | 34.0 %       |
| organic               | 28.4 %       |
| partner_store         | 23.8 %       |
| referral              | 13.9 %       |