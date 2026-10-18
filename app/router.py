"""
Endpoints del dashboard. Se montan en la misma API (service/main.py):
  GET /app                -> pagina del dashboard
  GET /app/replay         -> dias del replay listos para /predict
  GET /app/explanations   -> explicaciones de Granite pregeneradas (app/pregenerate.py)
"""

import json
from datetime import date
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.replay import observed_events, replay_days

APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "static" / "index.html"
EXPLANATIONS_PATH = APP_DIR / "explanations.json"

# Ventana por defecto: las dos semanas previas al apagon del 28-04-2025
DEFAULT_START = date(2025, 4, 14)
DEFAULT_END = date(2025, 4, 28)

router = APIRouter()


@router.get("/app", include_in_schema=False)
def dashboard():
    return FileResponse(INDEX_PATH)


@router.get("/app/replay")
def get_replay(start: date = DEFAULT_START, end: date = DEFAULT_END):
    if start > end:
        raise HTTPException(status_code=422, detail="start debe ser anterior a end")
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": replay_days(start, end),
        "events": observed_events(),
    }


@router.get("/app/explanations")
def get_explanations():
    if not EXPLANATIONS_PATH.exists():
        return {}
    return json.loads(EXPLANATIONS_PATH.read_text(encoding="utf-8"))
