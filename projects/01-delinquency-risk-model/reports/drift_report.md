# Drift monitoring report

Reference window: origination months 1-21. Monitored window: months 22-24 (includes the synthetic macro-shock).


## Population Stability Index by feature

|                                |    PSI | flag   |
|:-------------------------------|-------:|:-------|
| avg_prior_repayment_delay_days | 0.0072 | ok     |
| loan_to_income_ratio           | 0.0047 | ok     |
| credit_bureau_score            | 0.0046 | ok     |
| down_payment_ratio             | 0.0043 | ok     |
| monthly_income_mxn             | 0.0038 | ok     |
| num_active_loans_elsewhere     | 0.001  | ok     |


## Delinquency rate shift
- Reference: 13.55%
- Monitored: 19.06%


## Model performance on monitored window
- AUC (original held-out test set): 0.789
- AUC (monitored window, months 22-24): 0.789


## Calibration drift
- Mean predicted probability (monitored window): 15.43%
- Observed delinquency rate (monitored window): 19.06%
- Gap: 3.63%. The model under-predicts risk once the shock hits, even though PSI is clean. This is concept drift, not covariate drift. Outcome-rate monitoring catches it; PSI alone does not.
