import platform

from fastapi import FastAPI, HTTPException, status
from service.schemas import TelemetryInput, InferenceResponse
from service.utils import GridPredictor

from explain.schemas import ExplanationRequest, ExplanationResponse
from explain.generator import GraniteExplainer

app = FastAPI(
    title="GridZ Shield - Real-Time Risk Inference & Explanation API",
    version="1.0.0"
)

# Inicialización de servicios
try:
    predictor = GridPredictor()
except Exception as e:
    predictor = None

try:
    explainer = GraniteExplainer()
except Exception as e:
    explainer = None


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "ok", 
        "model_loaded": predictor is not None, 
        "explainer_loaded": explainer is not None,
        "arch": "s390x"
    }


@app.post("/predict", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
def predict_risk(telemetry: TelemetryInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo predictivo no disponible.")

    try:
        data = telemetry.model_dump()
        risk_score, level, top_factors = predictor.predict(data)
        return InferenceResponse(
            timestamp=telemetry.timestamp,
            risk_score=risk_score,
            level=level,
            top_factors=top_factors,
            is_simulated=telemetry.is_simulated or False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse, status_code=status.HTTP_200_OK)
def explain_risk(payload: ExplanationRequest):
    if explainer is None:
        raise HTTPException(status_code=503, detail="Servicio de explicabilidad no disponible.")

    try:
        return explainer.explain(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
