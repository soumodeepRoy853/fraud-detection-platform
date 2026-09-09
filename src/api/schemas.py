from pydantic import BaseModel, Field

class Transaction(BaseModel):
    account_id: str = Field(..., description="Unique account/card identifier")
    Time: float = Field(..., description="Seconds since the first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float



class ScoreResponse(BaseModel):
    fraud_probability: float
    risk_tier: str
    top_factors: list[dict]