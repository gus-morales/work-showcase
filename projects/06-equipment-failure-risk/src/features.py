"""Feature engineering for the failure-risk model. No missing values in
this dataset (telemetry signals are always available by construction),
so the pipeline here is lighter than project 01's: winsorize the
long-tailed sensor readings, one-hot encode the two categoricals. Built
as a feature-engine/sklearn Pipeline anyway, fit on the training split
only and reused unchanged elsewhere, for the same reason project 01/05
do it: a real deployment scores each truck-day against statistics
frozen at training time, not ones recomputed on data the model
shouldn't get to see yet.

`downtime_cost_usd` is deliberately excluded from the feature set: it's
an estimate of what an unplanned failure on that truck would cost, used
only in train.py's cost-optimal threshold search, not a signal that
should drive the failure-probability prediction itself."""
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline

CATEGORICAL_COLS = ["truck_class", "site"]

NUMERIC_COLS = [
    "engine_hours_at_obs", "equipment_age_years_at_obs", "vibration_rms_mm_s",
    "oil_pressure_psi", "coolant_temp_c", "brake_wear_pct", "fault_codes_7d",
    "days_since_last_pm", "payload_avg_tons", "ambient_temp_c",
]

WINSORIZE_COLS = ["vibration_rms_mm_s", "oil_pressure_psi", "coolant_temp_c", "brake_wear_pct", "payload_avg_tons"]

RAW_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def build_feature_pipeline() -> Pipeline:
    """Unfit sklearn Pipeline. Call .fit_transform() on the training
    split's raw feature columns, then .transform() (never re-fit) on
    every other split."""
    return Pipeline([
        ("winsorizer", Winsorizer(capping_method="iqr", tail="both", fold=1.5, variables=WINSORIZE_COLS)),
        ("onehot", OneHotEncoder(variables=CATEGORICAL_COLS, drop_last=False)),
    ])


def temporal_split(df: pd.DataFrame, train_end_day=420, val_end_day=510):
    """Same rationale as project 01/05: a random split would let the model
    implicitly see the future during training. train_end_day/val_end_day
    are day offsets from the dataset's epoch (2024-01-01), splitting the
    600-day window roughly 70/15/15."""
    epoch = pd.Timestamp("2024-01-01")
    day = (df["date"] - epoch).dt.days
    train = df[day <= train_end_day]
    val = df[(day > train_end_day) & (day <= val_end_day)]
    test = df[day > val_end_day]
    return train, val, test
