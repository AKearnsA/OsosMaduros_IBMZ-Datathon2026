from pydantic import BaseModel, Field
from typing import List


class ExplanationRequest(BaseModel):
    timestamp: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    level: str  # "Stable" | "Watch" | "Critical"
    top_factors: List[str]
    is_simulated: bool = False


class ExplanationResponse(BaseModel):
    summary: str
    recommended_actions: List[str]
    technical_details: str