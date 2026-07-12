"""Tests for the FastAPI scoring endpoint, using FastAPI's TestClient
against the real trained model artifact (reports/model.pkl). Skipped
automatically if that artifact hasn't been generated (it's gitignored
and produced by running src/train.py), the same way a real test suite
wouldn't fail CI over a missing model binary it doesn't own building."""
import pytest
from fastapi.testclient import TestClient

from serve import app, MODEL_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="reports/model.pkl not found; run src/train.py first"
)

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 34, "monthly_income_usd": 2400, "tenure_months_platform": 8,
    "num_previous_loans": 3, "credit_bureau_score": 690,
    "avg_prior_repayment_delay_days": 1.5, "num_active_loans_elsewhere": 1,
    "num_installments": 6, "loan_amount_usd": 850, "down_payment_ratio": 0.15,
    "city_tier": "tier1", "employment_type": "salaried", "device_type": "ios",
    "acquisition_channel": "organic", "merchant_category": "electronics",
}


def test_health_endpoint_reports_model_ready():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_path_exists": True}


def test_predict_returns_probability_and_decision():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["delinquency_probability"] <= 1.0
    assert body["decision"] in ("approve", "decline")
    assert 0.0 <= body["threshold"] <= 1.0


def test_predict_decision_matches_probability_vs_threshold():
    resp = client.post("/predict", json=VALID_PAYLOAD)
    body = resp.json()
    expected = "decline" if body["delinquency_probability"] >= body["threshold"] else "approve"
    assert body["decision"] == expected


def test_predict_accepts_missing_bureau_score():
    # About 9% of real applicants are thin-file; the field is optional for
    # exactly that reason and shouldn't 422.
    payload = {**VALID_PAYLOAD, "credit_bureau_score": None}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_riskier_profile_scores_higher():
    # Same applicant, but informal employment, thin credit history, and a
    # much smaller down payment, every one of these should push risk up
    # given the direction of every driver already established (README,
    # SHAP summary, survival hazard ratios). Not a tight bound, just a
    # sign check that the served pipeline matches the trained model's
    # known behavior rather than serving stale or mismatched artifacts.
    riskier_payload = {
        **VALID_PAYLOAD,
        "employment_type": "informal",
        "credit_bureau_score": 520,
        "down_payment_ratio": 0.0,
        "num_active_loans_elsewhere": 4,
        "avg_prior_repayment_delay_days": 15,
    }
    safe_prob = client.post("/predict", json=VALID_PAYLOAD).json()["delinquency_probability"]
    risky_prob = client.post("/predict", json=riskier_payload).json()["delinquency_probability"]
    assert risky_prob > safe_prob


@pytest.mark.parametrize("field,bad_value", [
    ("age", 10),                       # below ge=18
    ("monthly_income_usd", -100),      # not gt=0
    ("num_installments", 4),           # not in the allowed {3,6,9,12}
    ("down_payment_ratio", 1.5),       # above le=1
    ("city_tier", "tier9"),            # not a valid literal
    ("credit_bureau_score", 200),      # below ge=300
])
def test_predict_rejects_invalid_input(field, bad_value):
    payload = {**VALID_PAYLOAD, field: bad_value}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_required_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amount_usd"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
