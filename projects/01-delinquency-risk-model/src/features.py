"""Feature engineering for the delinquency model."""
import numpy as np
import pandas as pd

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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income_ratio"] = df["loan_amount_usd"] / df["monthly_income_usd"].clip(lower=1)
    df["installment_amount_usd"] = df["loan_amount_usd"] * (1 - df["down_payment_ratio"]) / df["num_installments"]
    df["installment_to_income_ratio"] = df["installment_amount_usd"] / df["monthly_income_usd"].clip(lower=1)
    df["low_bureau_score"] = (df["credit_bureau_score"] < 580).astype(int)
    df["is_new_customer"] = (df["tenure_months_platform"] < 3).astype(int)
    return df


def build_design_matrix(df: pd.DataFrame):
    """One-hot encode categoricals, return X (DataFrame) and feature name list."""
    num_cols = NUMERIC_COLS + ["installment_to_income_ratio", "low_bureau_score", "is_new_customer"]
    X_num = df[num_cols]
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    return X, list(X.columns)
