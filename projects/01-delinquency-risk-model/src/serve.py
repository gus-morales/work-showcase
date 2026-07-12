"""
A thin FastAPI wrapper around the trained model, the same shape a real
underwriting-time scoring service would take: take an applicant/loan
payload, run it through the frozen feature pipeline, return a
probability and an approve/decline decision at the cost-optimal
threshold picked in train.py. Not meant to demonstrate production
infrastructure (batching, auth, model versioning, canarying); it
demonstrates that the training artifacts are actually servable as-is,
not tied to notebook or script state.

Run:
    uvicorn serve:app --reload
Then:
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @example_request.json
"""
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from features import engineer_features, RAW_FEATURE_COLS

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "reports" / "model.pkl"

app = FastAPI(
    title="BNPL Delinquency Risk Model",
    description="Scores a loan application for 30+ days-past-due risk at origination.",
    version="1.0.0",
)

_bundle = None  # lazily loaded so importing this module doesn't require model.pkl to exist


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifact not found. Run src/train.py to produce reports/model.pkl.",
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class LoanApplication(BaseModel):
    """Raw applicant and loan fields, the same shape as a row of
    data/loans.csv before engineer_features derives ratio/flag columns.
    credit_bureau_score is optional: about 9% of real applicants are
    thin-file with no score on record, and the feature pipeline's
    missing-value handling (fit in train.py) is built to expect that."""

    age: int = Field(ge=18, le=100)
    monthly_income_usd: float = Field(gt=0)
    tenure_months_platform: float = Field(ge=0)
    num_previous_loans: int = Field(ge=0)
    credit_bureau_score: Optional[float] = Field(default=None, ge=300, le=850)
    avg_prior_repayment_delay_days: float = Field(ge=0)
    num_active_loans_elsewhere: int = Field(ge=0)
    num_installments: Literal[3, 6, 9, 12]
    loan_amount_usd: float = Field(gt=0, le=25_000)
    down_payment_ratio: float = Field(ge=0, le=1)
    city_tier: Literal["tier1", "tier2", "tier3"]
    employment_type: Literal["salaried", "self_employed", "gig_economy", "informal"]
    device_type: Literal["android", "ios", "web"]
    acquisition_channel: Literal["organic", "paid_social", "partner_store", "referral"]
    merchant_category: Literal["electronics", "fashion", "home_goods", "travel", "education", "groceries"]

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 34, "monthly_income_usd": 2400, "tenure_months_platform": 8,
            "num_previous_loans": 3, "credit_bureau_score": 690,
            "avg_prior_repayment_delay_days": 1.5, "num_active_loans_elsewhere": 1,
            "num_installments": 6, "loan_amount_usd": 850, "down_payment_ratio": 0.15,
            "city_tier": "tier1", "employment_type": "salaried", "device_type": "ios",
            "acquisition_channel": "organic", "merchant_category": "electronics",
        }
    })


class ScoreResponse(BaseModel):
    delinquency_probability: float
    decision: Literal["approve", "decline"]
    threshold: float


def score_application(payload: LoanApplication) -> ScoreResponse:
    bundle = get_bundle()
    row = pd.DataFrame([payload.model_dump()])
    row = engineer_features(row)
    X = bundle["feature_pipeline"].transform(row[RAW_FEATURE_COLS])
    prob = float(bundle["calibrated_model"].predict_proba(X)[:, 1][0])
    threshold = float(bundle["threshold"])
    decision = "decline" if prob >= threshold else "approve"
    return ScoreResponse(delinquency_probability=round(prob, 4), decision=decision, threshold=threshold)


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return {"status": "ok" if model_ready else "model_not_found", "model_path_exists": model_ready}


@app.post("/predict", response_model=ScoreResponse)
def predict(payload: LoanApplication):
    return score_application(payload)
