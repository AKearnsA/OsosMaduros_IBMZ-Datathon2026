import os
import platform

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from service.schemas import TelemetryInput, InferenceResponse
from service.utils import GridPredictor

from explain.schemas import ExplanationRequest, ExplanationResponse
from explain.generator import GraniteExplainer

app = FastAPI(
    title="GridZ Shield - Real-Time Risk Inference & Explanation API",
    version="1.0.0"
)

# Permite que el dashboard (app/) llame a la API desde el navegador.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Inicialización de servicios: si algo falla, se guarda el motivo para /health.
predictor, predictor_error = None, None
try:
    predictor = GridPredictor()
except Exception as e:
    predictor_error = f"{type(e).__name__}: {e}"
    print(f"[!] Modelo predictivo NO cargado -> {predictor_error}")

explainer, explainer_error = None, None
try:
    explainer = GraniteExplainer()
except Exception as e:
    explainer_error = f"{type(e).__name__}: {e}"
    print(f"[!] Explicador NO cargado -> {explainer_error}")


def llm_available() -> bool:
    """True solo si existen el binario de llama.cpp y el modelo Granite (si no, /explain usa reglas)."""
    return bool(
        explainer
        and explainer.binary_path
        and os.path.exists(explainer.model_path)
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_error": predictor_error,
        "explainer_loaded": explainer is not None,
        "explainer_error": explainer_error,
        "llm_available": llm_available(),
        "arch": platform.machine(),
    }


@app.post("/predict", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
def predict_risk(telemetry: TelemetryInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail=f"Modelo predictivo no disponible: {predictor_error}")

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
        raise HTTPException(status_code=503, detail=f"Servicio de explicabilidad no disponible: {explainer_error}")

    try:
        return explainer.explain(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
