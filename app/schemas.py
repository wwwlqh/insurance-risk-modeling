from pydantic import BaseModel, Field


class InsuranceInput(BaseModel):
    """
    Raw user input — same column values as the original CSV.
    Preprocessing (encoding, log transform, scaling) is handled server-side.
    """
    age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 45})
    gender: str = Field(..., json_schema_extra={"example": "Male"})
    marital_status: str = Field(..., json_schema_extra={"example": "Married"})
    urbanization_level: str = Field(..., json_schema_extra={"example": "Urban"})
    policy_term: int = Field(..., json_schema_extra={"example": 10})
    claim_frequency: int = Field(..., ge=0, json_schema_extra={"example": 2})
    maintenance_level: str = Field(..., json_schema_extra={"example": "Medium"})
    building_age: int = Field(..., ge=0, json_schema_extra={"example": 15})
    customer_satisfaction: int = Field(..., ge=1, le=5, json_schema_extra={"example": 4})
    has_security_system: str = Field(..., json_schema_extra={"example": "Yes"})
    construction_type: str = Field(..., json_schema_extra={"example": "Brick Wall"})
    policy_tenure: int = Field(..., ge=0, json_schema_extra={"example": 5})
    payment_method: str = Field(..., json_schema_extra={"example": "Online Payment"})
    credit_score: float = Field(..., json_schema_extra={"example": 650.0})
    fire_risk_score: float = Field(..., json_schema_extra={"example": 35.0})
    flood_risk_index: float = Field(..., json_schema_extra={"example": 40.0})
    crime_rate_index: float = Field(..., json_schema_extra={"example": 50.0})
    annual_income: float = Field(..., gt=0, json_schema_extra={"example": 50000.0})
    property_value: float = Field(..., gt=0, json_schema_extra={"example": 200000.0})
    premium_amount: float = Field(..., gt=0, json_schema_extra={"example": 1200.0})
    claim_amount_last: float = Field(..., ge=0, json_schema_extra={"example": 2000.0})


class ClassificationResponse(BaseModel):
    risk_category: int          # 0 = Low Risk, 1 = High Risk
    risk_label: str
    probability: float


class RegressionResponse(BaseModel):
    expected_claim_cost: float  # original dollar scale
