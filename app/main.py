"""
FastAPI app — two endpoints:
  POST /predict/classification  →  Risk_Category (0 Low / 1 High) + probability
  POST /predict/regression      →  Expected_Claim_Cost in original dollars
"""

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import InsuranceInput, ClassificationResponse, RegressionResponse
from app.preprocess import preprocess

# ── Load models once at startup ────────────────────────────────────────────────
try:
    _clf     = joblib.load("models/rf_classifier.joblib")
    _reg     = joblib.load("models/gb_regressor.joblib")
    _scaler_y = joblib.load("models/scaler_y.joblib")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model file not found: {e}. "
        "Run `python save_models.py` first to generate the models/ folder."
    )

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Insurance Risk Modeling API",
    description="Predict property insurance risk category and expected claim cost.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Insurance Risk API is running. Visit /docs for the interactive UI."}


@app.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(data: InsuranceInput):
    """Predict whether the property is Low Risk (0) or High Risk (1)."""
    try:
        X = preprocess(data)
        pred = int(_clf.predict(X)[0])
        prob = float(_clf.predict_proba(X)[0][1])
        return ClassificationResponse(
            risk_category=pred,
            risk_label="High Risk" if pred == 1 else "Low Risk",
            probability=round(prob, 4),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(data: InsuranceInput):
    """Predict the expected claim cost in original dollar scale."""
    try:
        X = preprocess(data)
        pred_scaled = _reg.predict(X)
        # Inverse pipeline: StandardScaler → inverse log1p
        cost = float(
            np.expm1(
                _scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
            ).flatten()[0]
        )
        return RegressionResponse(expected_claim_cost=round(cost, 2))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
