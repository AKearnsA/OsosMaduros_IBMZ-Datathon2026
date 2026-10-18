from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class TelemetryInput(BaseModel):
    """
    Una fila de data/features.parquet mas los historicos que calcula model/train.py.
    Los nombres coinciden con los del entrenamiento: el servicio no recalcula nada.
    """

    timestamp: datetime = Field(..., examples=["2025-04-25T22:00:00Z"])
    demand_mw: float = Field(..., gt=0)
    gen_total_mw: float = Field(..., gt=0)
    gen_nonsync_mw: float = Field(..., ge=0)
    gen_sync_mw: float = Field(..., ge=0)
    nonsync_share: float = Field(..., ge=0.0, le=1.0)
    interconnection_balance_mw: float = Field(...)
    inertia_est_gws: float = Field(..., gt=0)
    rocof_est_hz_s: float = Field(..., gt=0)
    # Historicos (dia anterior y media de 7 dias), como en model/train.py
    rocof_lag1: float = Field(..., gt=0)
    nonsync_share_lag1: float = Field(..., ge=0.0, le=1.0)
    rocof_roll7_mean: float = Field(..., gt=0)
    nonsync_roll7_mean: float = Field(..., ge=0.0, le=1.0)
    is_simulated: Optional[bool] = Field(False)


class InferenceResponse(BaseModel):
    timestamp: datetime
    risk_score: float = Field(..., ge=0.0, le=1.0)
    level: str  # "Stable" | "Watch" | "Critical"
    top_factors: List[str]
    is_simulated: bool
