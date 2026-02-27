"""
Streamlit dashboard for Insurance Risk Modeling.
Sends requests to the FastAPI backend at API_URL.
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Insurance Risk Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Insurance Risk Predictor")
st.markdown("Fill in the property and policyholder details below, then click **Predict**.")

# ── Sidebar — input form ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Input Features")

    age                  = st.slider("Age", 18, 100, 45)
    gender               = st.selectbox("Gender", ["Male", "Female"])
    marital_status       = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    urbanization_level   = st.selectbox("Urbanization Level", ["Urban", "Suburban", "Rural"])
    policy_term          = st.selectbox("Policy Term (years)", [5, 10, 15, 20])
    claim_frequency      = st.slider("Claim Frequency", 0, 5, 2)
    maintenance_level    = st.selectbox("Maintenance Level", ["Low", "Moderate", "High"])
    building_age         = st.slider("Building Age (years)", 0, 100, 15)
    customer_satisfaction = st.slider("Customer Satisfaction (1–5)", 1, 5, 4)
    has_security_system  = st.selectbox("Has Security System", ["Yes", "No"])
    construction_type    = st.selectbox("Construction Type", [
        "Brick Wall", "Lightweight Concrete", "Reinforced Concrete", "Steel Frame", "Timber Structure"
    ])
    policy_tenure        = st.slider("Policy Tenure (years)", 0, 20, 5)
    payment_method       = st.selectbox("Payment Method", [
        "Cash", "Online Payment", "Mortgage", "Financing Scheme", "Bank Loan"
    ])

st.subheader("Financial & Risk Details")
col1, col2, col3 = st.columns(3)

with col1:
    annual_income      = st.number_input("Annual Income (RM)", min_value=0.0, value=50000.0, step=1000.0)
    property_value     = st.number_input("Property Value (RM)", min_value=0.0, value=200000.0, step=5000.0)
    premium_amount     = st.number_input("Premium Amount (RM)", min_value=0.0, value=1200.0, step=100.0)

with col2:
    claim_amount_last  = st.number_input("Last Claim Amount (RM)", min_value=0.0, value=2000.0, step=500.0)
    credit_score       = st.number_input("Credit Score", min_value=300.0, max_value=850.0, value=650.0)
    fire_risk_score    = st.number_input("Fire Risk Score", min_value=0.0, max_value=100.0, value=35.0)

with col3:
    flood_risk_index   = st.number_input("Flood Risk Index", min_value=0.0, max_value=100.0, value=40.0)
    crime_rate_index   = st.number_input("Crime Rate Index", min_value=0.0, max_value=100.0, value=50.0)

# ── Predict button ─────────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Predict", type="primary", use_container_width=True):

    payload = {
        "age":                    age,
        "gender":                 gender,
        "marital_status":         marital_status,
        "urbanization_level":     urbanization_level,
        "policy_term":            policy_term,
        "claim_frequency":        claim_frequency,
        "maintenance_level":      maintenance_level,
        "building_age":           building_age,
        "customer_satisfaction":  float(customer_satisfaction),
        "has_security_system":    has_security_system,
        "construction_type":      construction_type,
        "policy_tenure":          policy_tenure,
        "payment_method":         payment_method,
        "credit_score":           credit_score,
        "fire_risk_score":        fire_risk_score,
        "flood_risk_index":       flood_risk_index,
        "crime_rate_index":       crime_rate_index,
        "annual_income":          annual_income,
        "property_value":         property_value,
        "premium_amount":         premium_amount,
        "claim_amount_last":      claim_amount_last,
    }

    try:
        clf_res = requests.post(f"{API_URL}/predict/classification", json=payload, timeout=10)
        reg_res = requests.post(f"{API_URL}/predict/regression",     json=payload, timeout=10)
        clf_res.raise_for_status()
        reg_res.raise_for_status()

        clf_data = clf_res.json()
        reg_data = reg_res.json()

        col_a, col_b = st.columns(2)

        with col_a:
            label = clf_data["risk_label"]
            prob  = clf_data["probability"]
            color = "🔴" if clf_data["risk_category"] == 1 else "🟢"
            st.metric(f"{color} Risk Category", label)
            st.progress(prob, text=f"High-Risk Probability: {prob:.1%}")

        with col_b:
            cost = reg_data["expected_claim_cost"]
            st.metric("💰 Expected Claim Cost", f"RM {cost:,.2f}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("Insurance Risk Modeling Project | FastAPI + Streamlit")
