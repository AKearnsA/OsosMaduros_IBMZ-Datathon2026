from pydantic import BaseModel
from typing import List


class ExplanationRequest(BaseModel):
    timestamp: str
    risk_score: float
    level: str
    top_factors: List[str]
    is_simulated: bool = False


class ExplanationResponse(BaseModel):
    summary: str
    recommended_actions: List[str]
    technical_details: str