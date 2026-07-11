"""Feature engineering for the delinquency model.

Ratio/flag features are still plain pandas (cheap, deterministic, no
fitting required). Everything that needs to be *fit* on training data
only (missing-value imputation, outlier capping, categorical encoding)
is built as a feature-engine pipeline instead, so it can be fit once on
the training split and reused, unchanged, on validation/test/monitoring
data, the same way a real deployment would freeze its preprocessing
statistics at training time rather than recomputing them on new data.
"""
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline

CATEGORICAL_COLS = [
    "city_tier", "employment_type", "device_type",
    "acquisition_channel", "merchant_category",
]

NUMERIC_COLS = [
    "age", "monthly_income_usd", "tenure_months_platform", "num_previous_loans",
    "credit_bureau_score", "avg_prior_repayment_delay_days", "num_active_loans_elsewhere",
    "num_installments", "loan_amount_usd", "down_payment_ratio",
    "loan_to_income_ratio", "installment_amount_usd",
]

# credit_bureau_score is missing for a share of thin-file customers (see
# generate_data.py); everything else is complete by construction, so the
# imputer is scoped to just this column rather than applied blindly.
MISSING_COLS = ["credit_bureau_score"]

# Long-tailed dollar amounts, capped rather than left to swing the model
# on a handful of extreme originations.
WINSORIZE_COLS = ["loan_amount_usd", "monthly_income_usd"]

ENGINEERED_NUMERIC_COLS = ["installment_to_income_ratio", "low_bureau_score", "is_new_customer"]
RAW_FEATURE_COLS = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS + CATEGORICAL_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income_ratio"] = df["loan_amount_usd"] / df["monthly_income_usd"].clip(lower=1)
    df["installment_amount_usd"] = df["loan_amount_usd"] * (1 - df["down_payment_ratio"]) / df["num_installments"]
    df["installment_to_income_ratio"] = df["installment_amount_usd"] / df["monthly_income_usd"].clip(lower=1)
    # NaN bureau scores compare False here (not < 580), which is the right
    # behavior: "unknown" and "known-low" are different states, and
    # "unknown" gets its own explicit signal from the pipeline's missing
    # indicator rather than being folded into this flag.
    df["low_bureau_score"] = (df["credit_bureau_score"] < 580).astype(int)
    df["is_new_customer"] = (df["tenure_months_platform"] < 3).astype(int)
    return df


def build_feature_pipeline() -> Pipeline:
    """Unfit sklearn Pipeline. Call .fit_transform() on the training
    split's raw feature columns, then .transform() (never re-fit) on
    every other split or monitoring window."""
    return Pipeline([
        ("missing_indicator", AddMissingIndicator(variables=MISSING_COLS)),
        ("bureau_imputer", MeanMedianImputer(imputation_method="median", variables=MISSING_COLS)),
        ("winsorizer", Winsorizer(capping_method="iqr", tail="both", fold=1.5, variables=WINSORIZE_COLS)),
        ("onehot", OneHotEncoder(variables=CATEGORICAL_COLS, drop_last=False)),
    ])
