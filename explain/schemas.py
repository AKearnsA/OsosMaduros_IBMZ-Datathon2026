from pydantic import BaseModel, Field
from typing import List, Optional


class ExplanationRequest(BaseModel):
    # Resultado de /predict (contrato de ARCHITECTURE.md)
    timestamp: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    level: str  # "Stable" | "Watch" | "Critical"
    top_factors: List[str]
    is_simulated: bool = False
    # Valores del dia (opcionales): dan a Granite cifras reales para que no las invente
    inertia_est_gws: Optional[float] = None
    nonsync_share: Optional[float] = None
    interconnection_balance_mw: Optional[float] = None
    rocof_est_hz_s: Optional[float] = None
    demand_mw: Optional[float] = None


class ExplanationResponse(BaseModel):
    summary: str
    recommended_actions: List[str]
    technical_details: str
    source: str = "rules"  # "granite" si lo ha redactado el LLM, "rules" si es el respaldo
