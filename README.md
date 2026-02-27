# Insurance Risk Modeling

A machine learning system that predicts property insurance risk and expected claim costs, served via a FastAPI REST API and an interactive Streamlit dashboard — all containerized with Docker.

---

## Project Overview

| Model | Task | Algorithm |
|-------|------|-----------|
| Risk Classifier | Predict Low Risk (0) / High Risk (1) | Random Forest |
| Cost Regressor | Predict expected claim cost (RM) | Gradient Boosting |

**Input:** 21 property and policyholder features (demographics, property attributes, risk scores, financial data)

**Output:**
- Risk category + probability (classification)
- Expected claim cost in original dollar scale (regression)

---

## Project Structure

```
Insurance Risk Modeling/
├── app/                        # FastAPI backend
│   ├── main.py                 # API endpoints
│   ├── schemas.py              # Pydantic request/response models
│   └── preprocess.py           # Feature engineering pipeline
├── streamlit_app/
│   └── app.py                  # Streamlit dashboard
├── models/                     # Saved model artifacts (joblib)
│   ├── rf_classifier.joblib
│   ├── gb_regressor.joblib
│   ├── label_encoders.joblib
│   ├── scaler_X.joblib
│   ├── scaler_y.joblib
│   ├── scale_features.joblib
│   └── feature_order.joblib
├── tests/
│   └── test_api.py             # Pytest API tests
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD
├── notebook.ipynb              # Full ML pipeline notebook
├── Dockerfile                  # API container
├── Dockerfile.streamlit        # Streamlit container
├── docker-compose.yml          # Orchestration
├── requirements.txt            # API dependencies
├── requirements-streamlit.txt  # Streamlit dependencies
└── test_models.py              # Model sanity check script
```

---

## Quick Start (Docker)

### Run everything with one command

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI (interactive docs) | http://localhost:8000/docs |
| FastAPI (root) | http://localhost:8000 |

### Stop containers

```bash
docker compose down
```

---

## API Endpoints

### `POST /predict/classification`
Predicts whether a property is **Low Risk** or **High Risk**.

**Response:**
```json
{
  "risk_category": 1,
  "risk_label": "High Risk",
  "probability": 0.8234
}
```

### `POST /predict/regression`
Predicts the **expected claim cost** in original dollar scale.

**Response:**
```json
{
  "expected_claim_cost": 4521.75
}
```

### Sample Request Payload

```json
{
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
  "claim_amount_last": 2000.0
}
```

---

## Run Tests

```bash
pip install -r requirements.txt pytest httpx
pytest tests/ -v
```

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every push/PR to `main`:

1. **Test** — runs `pytest tests/` with Python 3.10
2. **Build** — builds both Docker images (API + Streamlit)
3. **Push** *(optional)* — push images to Docker Hub (requires `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets)

To enable Docker Hub push, add secrets in:
> GitHub repo → Settings → Secrets and variables → Actions

---

## Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload --port 8000

# Start Streamlit (separate terminal)
pip install -r requirements-streamlit.txt
streamlit run streamlit_app/app.py
```

---

## Dataset

- `Property Insurance.csv` — policyholder and property features
- `External Variables.csv` — external risk indices (flood, crime, fire)
