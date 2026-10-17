from fastapi import FastAPI, status
from explain.schemas import ExplanationRequest, ExplanationResponse
from explain.generator import RiskExplainer

# Si se agrega en el main existente:
explainer = RiskExplainer()

@app.post("/explain", response_model=ExplanationResponse, status_code=status.HTTP_200_OK)
def explain_risk(payload: ExplanationRequest):
    return explainer.explain(payload)