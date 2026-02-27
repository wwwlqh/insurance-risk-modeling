"""
Quick sanity check — run this to verify all saved joblib files work.
Usage: python test_models.py
"""

import joblib
import numpy as np
import pandas as pd

print("=" * 50)
print("1. Loading saved artifacts...")
print("=" * 50)

rf             = joblib.load("models/rf_classifier.joblib")
gb             = joblib.load("models/gb_regressor.joblib")
scaler_X       = joblib.load("models/scaler_X.joblib")
scaler_y       = joblib.load("models/scaler_y.joblib")
label_encoders = joblib.load("models/label_encoders.joblib")
scale_features = joblib.load("models/scale_features.joblib")
feature_order  = joblib.load("models/feature_order.joblib")

print("All files loaded successfully.")
print(f"  Feature order ({len(feature_order)} cols): {feature_order}")
print(f"  Scale features ({len(scale_features)}): {scale_features}")
print(f"  Label encoders: {list(label_encoders.keys())}")

# ── Build a sample input (raw values, same as what user would type) ───────────
print("\n" + "=" * 50)
print("2. Building sample raw input...")
print("=" * 50)

raw = {
    'Age':                   45,
    'Gender':                'Male',
    'Marital_Status':        'Married',
    'Urbanization_Level':    'Urban',
    'Policy_Term':           10,
    'Claim_Frequency':       2,
    'Maintenance_Level':     'Moderate',
    'Building_Age':          15,
    'Customer_Satisfaction': 4.0,   # must be float — encoder stores '4.0' not '4'
    'Has_Security_System':   'Yes',
    'Construction_Type':     'Brick Wall',
    'Policy_Tenure':         5,
    'Payment_Method':        'Online Payment',
    'Credit_Score':          650.0,
    'Fire_Risk_Score':       35.0,
    # These 2 get combined into Area_Risk_Index
    'Flood_Risk_Index':      40.0,
    'Crime_Rate_Index':      50.0,
    # These get log1p transformed
    'Annual_Income':         50000.0,
    'Property_Value':        200000.0,
    'Premium_Amount':        1200.0,
    'Claim_Amount_Last':     2000.0,
}

# ── Preprocess (mirrors app/preprocess.py) ────────────────────────────────────
print("\n" + "=" * 50)
print("3. Preprocessing...")
print("=" * 50)

ENCODE_COLS = [
    'Gender', 'Marital_Status', 'Urbanization_Level', 'Policy_Term',
    'Claim_Frequency', 'Maintenance_Level', 'Customer_Satisfaction',
    'Has_Security_System', 'Construction_Type', 'Payment_Method',
]

row = {
    'Age':                   raw['Age'],
    'Gender':                raw['Gender'],
    'Marital_Status':        raw['Marital_Status'],
    'Urbanization_Level':    raw['Urbanization_Level'],
    'Policy_Term':           raw['Policy_Term'],
    'Claim_Frequency':       raw['Claim_Frequency'],
    'Maintenance_Level':     raw['Maintenance_Level'],
    'Building_Age':          raw['Building_Age'],
    'Customer_Satisfaction': raw['Customer_Satisfaction'],
    'Has_Security_System':   raw['Has_Security_System'],
    'Construction_Type':     raw['Construction_Type'],
    'Policy_Tenure':         raw['Policy_Tenure'],
    'Payment_Method':        raw['Payment_Method'],
    'Credit_Score':          raw['Credit_Score'],
    'Fire_Risk_Score':       raw['Fire_Risk_Score'],
    'Area_Risk_Index':       (raw['Flood_Risk_Index'] + raw['Crime_Rate_Index']) / 2,
    'log_Annual_Income':     np.log1p(raw['Annual_Income']),
    'log_Property_Value':    np.log1p(raw['Property_Value']),
    'log_Premium_Amount':    np.log1p(raw['Premium_Amount']),
    'log_Claim_Amount_Last': np.log1p(raw['Claim_Amount_Last']),
}

# Label encode (Customer_Satisfaction stored as float string '4.0' in encoder)
for col in ENCODE_COLS:
    le = label_encoders.get(col)
    if le:
        val = str(row[col])
        if val not in le.classes_:
            val = str(float(row[col]))
        row[col] = int(le.transform([val])[0])

print("  After encoding (categorical cols):")
for col in ENCODE_COLS:
    print(f"    {col}: {row[col]}")

# Build DataFrame in correct column order
df = pd.DataFrame([row])[feature_order]

# Scale
df[scale_features] = scaler_X.transform(df[scale_features])
X = df.values

print(f"\n  Final feature array shape: {X.shape}")

# ── Predict ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("4. Predictions")
print("=" * 50)

# Classification
clf_pred = int(rf.predict(X)[0])
clf_prob = float(rf.predict_proba(X)[0][1])
print(f"  Risk Category : {clf_pred} ({'High Risk' if clf_pred == 1 else 'Low Risk'})")
print(f"  Probability   : {clf_prob:.4f}")

# Regression
reg_pred_scaled = gb.predict(X)
cost = float(np.expm1(scaler_y.inverse_transform(reg_pred_scaled.reshape(-1, 1))).flatten()[0])
print(f"  Expected Cost : RM {cost:,.2f}")

print("\n" + "=" * 50)
print("All checks passed. Models are working correctly.")
print("=" * 50)
