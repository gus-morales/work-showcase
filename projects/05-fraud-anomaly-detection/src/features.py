"""Feature engineering for the fraud model. No missing values in this
dataset (checkout-time signals are always available by construction),
so the pipeline is light: winsorize the long-tailed dollar amount,
one-hot encode the two categoricals. Built as a feature-engine/sklearn
Pipeline anyway, fit on the training split only and reused unchanged
elsewhere, since a real deployment scores each transaction against
statistics frozen at training time, not ones recomputed on data the
model shouldn't get to see yet."""
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline

CATEGORICAL_COLS = ["merchant_category", "device_type"]

NUMERIC_COLS = [
    "account_age_days_at_tx", "amount_usd", "checkout_seconds",
    "transactions_last_1h", "transactions_last_24h",
    "is_new_device", "billing_shipping_mismatch", "ip_billing_country_mismatch",
]

WINSORIZE_COLS = ["amount_usd", "checkout_seconds"]

RAW_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def build_feature_pipeline() -> Pipeline:
    """Unfit sklearn Pipeline. Call .fit_transform() on the training
    split's raw feature columns, then .transform() (never re-fit) on
    every other split."""
    return Pipeline([
        ("winsorizer", Winsorizer(capping_method="iqr", tail="both", fold=1.5, variables=WINSORIZE_COLS)),
        ("onehot", OneHotEncoder(variables=CATEGORICAL_COLS, drop_last=False)),
    ])


def temporal_split(df: pd.DataFrame, train_end_day=126, val_end_day=153):
    """A random split would let the model implicitly see the future
    during training, so the split follows time instead. train_end_day/val_end_day
    are day offsets from the dataset's epoch (2025-01-01), splitting the
    180-day window roughly 70/15/15."""
    epoch = pd.Timestamp("2025-01-01")
    day = (df["timestamp"] - epoch).dt.days
    train = df[day <= train_end_day]
    val = df[(day > train_end_day) & (day <= val_end_day)]
    test = df[day > val_end_day]
    return train, val, test
