"""
Mock all joblib.load calls so CI tests never touch actual model files.
This conftest.py is loaded before any test module, so the patch is
active when `from app.main import app` triggers the module-level loads.
"""

import numpy as np
from unittest.mock import MagicMock
import joblib

# ── Feature metadata (must match what preprocess.py expects) ─────────────────
FEATURE_ORDER = [
    'Age', 'Gender', 'Marital_Status', 'Urbanization_Level', 'Policy_Term',
    'Claim_Frequency', 'Maintenance_Level', 'Building_Age',
    'Customer_Satisfaction', 'Has_Security_System', 'Construction_Type',
    'Policy_Tenure', 'Payment_Method', 'Credit_Score', 'Fire_Risk_Score',
    'Area_Risk_Index', 'log_Annual_Income', 'log_Property_Value',
    'log_Premium_Amount', 'log_Claim_Amount_Last',
]

SCALE_FEATURES = [
    'Age', 'Building_Age', 'Policy_Tenure', 'Credit_Score',
    'Fire_Risk_Score', 'Area_Risk_Index', 'log_Annual_Income',
    'log_Property_Value', 'log_Premium_Amount', 'log_Claim_Amount_Last',
]

ENCODE_COLS = [
    'Gender', 'Marital_Status', 'Urbanization_Level', 'Policy_Term',
    'Claim_Frequency', 'Maintenance_Level', 'Customer_Satisfaction',
    'Has_Security_System', 'Construction_Type', 'Payment_Method',
]

# ── Mock label encoders ───────────────────────────────────────────────────────
def _make_le(classes):
    le = MagicMock()
    le.classes_ = np.array(classes)
    le.transform = lambda vals: np.array([0] * len(vals))
    return le

mock_label_encoders = {
    'Gender':               _make_le(['Female', 'Male']),
    'Marital_Status':       _make_le(['Divorced', 'Married', 'Single', 'Widowed']),
    'Urbanization_Level':   _make_le(['Rural', 'Suburban', 'Urban']),
    'Policy_Term':          _make_le(['5', '10', '15', '20']),
    'Claim_Frequency':      _make_le(['0', '1', '2', '3', '4', '5']),
    'Maintenance_Level':    _make_le(['High', 'Low', 'Moderate']),
    'Customer_Satisfaction': _make_le(['1.0', '2.0', '3.0', '4.0', '5.0']),
    'Has_Security_System':  _make_le(['No', 'Yes']),
    'Construction_Type':    _make_le(['Brick Wall', 'Lightweight Concrete',
                                      'Reinforced Concrete', 'Steel Frame', 'Timber Structure']),
    'Payment_Method':       _make_le(['Bank Loan', 'Cash', 'Financing Scheme',
                                      'Mortgage', 'Online Payment']),
}

# ── Mock scaler (returns zeros, shape-preserving) ─────────────────────────────
mock_scaler_X = MagicMock()
mock_scaler_X.transform = lambda df: np.zeros((len(df), len(SCALE_FEATURES)))

mock_scaler_y = MagicMock()
mock_scaler_y.inverse_transform = lambda x: np.array([[8.5]])

# ── Mock models ───────────────────────────────────────────────────────────────
mock_clf = MagicMock()
mock_clf.predict = lambda X: np.array([1])
mock_clf.predict_proba = lambda X: np.array([[0.2, 0.8]])

mock_reg = MagicMock()
mock_reg.predict = lambda X: np.array([1.5])

# ── Patch joblib.load ─────────────────────────────────────────────────────────
def _mock_load(path, *args, **kwargs):
    p = str(path)
    if 'rf_classifier'   in p: return mock_clf
    if 'gb_regressor'    in p: return mock_reg
    if 'scaler_y'        in p: return mock_scaler_y
    if 'scaler_X'        in p: return mock_scaler_X
    if 'label_encoders'  in p: return mock_label_encoders
    if 'scale_features'  in p: return SCALE_FEATURES
    if 'feature_order'   in p: return FEATURE_ORDER
    return MagicMock()

joblib.load = _mock_load
