"""
Mirrors the preprocessing pipeline from notebook.ipynb:
  1. Label encode categorical features  (same LabelEncoders, fitted with .astype(str))
  2. Combine flood + crime → Area_Risk_Index
  3. log1p transform skewed columns
  4. Reorder columns to match X_train
  5. StandardScaler on numerical features only
"""

import numpy as np
import pandas as pd
import joblib

# ── Load saved artifacts once at import time ─────────────────────────────────
_scaler_X       = joblib.load("models/scaler_X.joblib")
_label_encoders = joblib.load("models/label_encoders.joblib")
_scale_features = joblib.load("models/scale_features.joblib")
_feature_order  = joblib.load("models/feature_order.joblib")

# Categorical columns that need label encoding (exclude targets + dropped cols)
_ENCODE_COLS = [
    'Gender', 'Marital_Status', 'Urbanization_Level', 'Policy_Term',
    'Claim_Frequency', 'Maintenance_Level', 'Customer_Satisfaction',
    'Has_Security_System', 'Construction_Type', 'Payment_Method',
]


def preprocess(data) -> np.ndarray:
    """
    Convert a raw InsuranceInput into a scaled numpy array ready for prediction.
    """
    row = {
        'Age':                   data.age,
        'Gender':                data.gender,
        'Marital_Status':        data.marital_status,
        'Urbanization_Level':    data.urbanization_level,
        'Policy_Term':           data.policy_term,
        'Claim_Frequency':       data.claim_frequency,
        'Maintenance_Level':     data.maintenance_level,
        'Building_Age':          data.building_age,
        'Customer_Satisfaction': data.customer_satisfaction,
        'Has_Security_System':   data.has_security_system,
        'Construction_Type':     data.construction_type,
        'Policy_Tenure':         data.policy_tenure,
        'Payment_Method':        data.payment_method,
        'Credit_Score':          data.credit_score,
        'Fire_Risk_Score':       data.fire_risk_score,
        # Combine risk indices (mirrors notebook cell 7b919fb5)
        'Area_Risk_Index':       (data.flood_risk_index + data.crime_rate_index) / 2,
        # Log transform (mirrors notebook cell 7eae4b67)
        'log_Annual_Income':     np.log1p(data.annual_income),
        'log_Property_Value':    np.log1p(data.property_value),
        'log_Premium_Amount':    np.log1p(data.premium_amount),
        'log_Claim_Amount_Last': np.log1p(data.claim_amount_last),
    }

    # Label encode — must match .astype(str) used during notebook fitting
    # Customer_Satisfaction was stored as float in notebook (e.g. 4 → '4.0')
    for col in _ENCODE_COLS:
        le = _label_encoders.get(col)
        if le is None:
            continue
        val = str(row[col])
        if val not in le.classes_:
            val = str(float(row[col]))  # try float string e.g. '4' → '4.0'
        try:
            row[col] = int(le.transform([val])[0])
        except ValueError:
            row[col] = 0  # unseen label: fall back to most common class

    # Build DataFrame in the exact column order X_train used
    df = pd.DataFrame([row])[_feature_order]

    # Scale numerical features
    df[_scale_features] = _scaler_X.transform(df[_scale_features])

    return df  # return DataFrame so model gets feature names (suppresses sklearn warning)
