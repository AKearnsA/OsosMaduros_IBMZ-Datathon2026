"""
Datos para el replay historico del dashboard.

Cada dia del replay es una fila de data/features.parquet preparada exactamente igual
que en model/train.py (mismos historicos), lista para enviarla a /predict.
La prediccion hecha con los datos del dia D es el riesgo del dia D+1.
"""

from datetime import date, timedelta
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from model.train import DATA_PATH, load_and_prepare_dataset

# Variables que espera /predict (service/schemas.py), con los nombres de train.py
FEATURES = [
    "demand_mw",
    "gen_total_mw",
    "gen_nonsync_mw",
    "gen_sync_mw",
    "nonsync_share",
    "interconnection_balance_mw",
    "inertia_est_gws",
    "rocof_est_hz_s",
    "rocof_lag1",
    "nonsync_share_lag1",
    "rocof_roll7_mean",
    "nonsync_roll7_mean",
]


def _local_dates(timestamps: pd.Series) -> pd.Series:
    """Fecha en hora peninsular (el timestamp UTC de cada fila es la medianoche local)."""
    try:
        return timestamps.dt.tz_convert("Europe/Madrid").dt.date
    except Exception:
        # Sin base de datos de zonas horarias: la medianoche local cae entre 22:00 y 23:00 UTC
        return (timestamps + pd.Timedelta(hours=2)).dt.date


@lru_cache(maxsize=1)
def _prepared() -> pd.DataFrame:
    df = load_and_prepare_dataset(DATA_PATH)
    df["date"] = _local_dates(df["timestamp"])
    df["forecast_date"] = df["date"] + timedelta(days=1)
    return df


@lru_cache(maxsize=1)
def observed_events() -> List[Dict]:
    """Eventos reales documentados (label_event == 1) para marcarlos en el grafico."""
    raw = pd.read_parquet(DATA_PATH)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    events = raw[raw["label_event"] == 1]
    return [
        {"date": d.isoformat(), "label": "Apagon iberico (12:33)"}
        for d in _local_dates(events["timestamp"])
    ]


def replay_days(start: date, end: date) -> List[Dict]:
    """
    Dias del replay cuya prevision cae entre start y end (ambos incluidos).
    El dia del apagon no tiene fila propia: train.py lo excluye por estar
    distorsionado por el propio apagon.
    """
    df = _prepared()
    actual_by_date = dict(zip(df["date"], df["rocof_est_hz_s"]))
    window = df[(df["forecast_date"] >= start) & (df["forecast_date"] <= end)]

    days = []
    for _, row in window.iterrows():
        actual = actual_by_date.get(row["forecast_date"])
        days.append(
            {
                "date": row["date"].isoformat(),
                "forecast_date": row["forecast_date"].isoformat(),
                "timestamp": row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "features": {f: float(row[f]) for f in FEATURES},
                "actual_rocof_next_day": None if actual is None else float(actual),
            }
        )
    return days
