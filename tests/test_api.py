"""
Basic API tests — run with: pytest tests/
Requires models/ folder to exist (run save_models.py first).
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "age": 45,
    "gender": "Male",
    "marital_status": "Married",
    "urbanization_level": "Urban",
    "policy_term": 10,
    "claim_frequency": 2,
    "maintenance_level": "Medium",
    "building_age": 15,
    "customer_satisfaction": 4,
    "has_security_system": "Yes",
    "construction_type": "Brick Wall",
    "policy_tenure": 5,
    "payment_method": "Online Payment",
    "credit_score": 650.0,
    "fire_risk_score": 35.0,
    "flood_risk_index": 40.0,
    "crime_rate_index": 50.0,
    "annual_income": 50000.0,
    "property_value": 200000.0,
    "premium_amount": 1200.0,
    "claim_amount_last": 2000.0,
}


def test_root():
    res = client.get("/")
    assert res.status_code == 200


def test_classification():
    res = client.post("/predict/classification", json=SAMPLE_PAYLOAD)
    assert res.status_code == 200
    data = res.json()
    assert "risk_category" in data
    assert data["risk_category"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0


def test_regression():
    res = client.post("/predict/regression", json=SAMPLE_PAYLOAD)
    assert res.status_code == 200
    data = res.json()
    assert "expected_claim_cost" in data
    assert data["expected_claim_cost"] > 0
