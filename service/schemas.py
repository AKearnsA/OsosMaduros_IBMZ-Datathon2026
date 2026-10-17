from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    timestamp: datetime = Field(..., example="2025-04-28T10:30:00Z")
    demand_mw: float = Field(..., gt=0)
    gen_total_mw: float = Field(..., gt=0)
    gen_nonsync_mw: float = Field(..., ge=0)
    gen_sync_mw: float = Field(..., ge=0)
    interconnection_balance_mw: float = Field(...)
    frequency_hz: Optional[float] = Field(None)
    rocof_lag_1d: Optional[float] = Field(0.0)
    rocof_ma_7d: Optional[float] = Field(0.0)
    is_simulated: Optional[bool] = Field(False)


class InferenceResponse(BaseModel):
    timestamp: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0)
    level: str  # "Stable" | "Watch" | "Critical"
    top_factors: List[str]
    is_simulated: bool
    