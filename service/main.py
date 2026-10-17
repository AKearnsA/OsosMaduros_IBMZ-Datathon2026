from fastapi import FastAPI, HTTPException, status
from service.schemas import TelemetryInput, InferenceResponse
from service.utils import GridPredictor

app = FastAPI(
    title="GridZ Shield - Real-Time Risk Inference API",
    version="1.0.0"
)

try:
    predictor = GridPredictor()
except Exception as e:
    predictor = None

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None, "arch": "s390x"}

@app.post("/predict", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
def predict_risk(telemetry: TelemetryInput):
    if predictor is None:
        raise HTTPException(status_code=533, detail="Modelo no disponible.")
    
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
        