"""
Módulo de Ingesta de Datos y Conectores para GridZ Shield.
Compatible con arquitectura IBM LinuxONE (s390x).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
import numpy as np
import pandas as pd
import requests

DATASET_PATH = "data/features.parquet"


def fetch_redata_structure(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Obtiene la estructura de generación de la API pública de REData (Red Eléctrica).
    """
    url = (
        f"https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
        f"?start_date={start_date}T00:00:00&end_date={end_date}T23:59:59&time_trunc=hour&geo_limit=peninsular"
    )
    headers = {
        "Accept": "application/json",
        "User-Agent": "GridZShield/1.0",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        included = data.get("included", [])
        records = []

        for tech in included:
            type_name = tech.get("type", "")
            values = tech.get("attributes", {}).get("values", [])
            for v in values:
                records.append(
                    {
                        "timestamp": v["datetime"],
                        "tech": type_name,
                        "value_mw": v["value"],
                    }
                )

        if not records:
            return None

        df_raw = pd.DataFrame(records)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)

        df_pivoted = df_raw.pivot_table(
            index="timestamp", columns="tech", values="value_mw", aggfunc="sum"
        ).fillna(0.0)

        nonsync_cols = [
            c
            for c in df_pivoted.columns
            if any(k in c.lower() for k in ["eólica", "solar", "wind", "pv"])
        ]
        sync_cols = [c for c in df_pivoted.columns if c not in nonsync_cols]

        df_features = pd.DataFrame(index=df_pivoted.index)
        df_features["gen_nonsync_mw"] = (
            df_pivoted[nonsync_cols].sum(axis=1) if nonsync_cols else 0.0
        )
        df_features["gen_sync_mw"] = (
            df_pivoted[sync_cols].sum(axis=1) if sync_cols else 10000.0
        )
        df_features["gen_total_mw"] = (
            df_features["gen_nonsync_mw"] + df_features["gen_sync_mw"]
        )
        df_features["demand_mw"] = df_features["gen_total_mw"] * 0.98
        df_features["interconnection_balance_mw"] = 0.0
        df_features["frequency_hz"] = 50.0
        df_features["rocof_hz_s"] = 0.0
        df_features["label_event"] = 0

        return df_features.reset_index()

    except Exception as e:
        print(f"[!] Excepción al conectar con la API de REData: {e}")
        return None


def generate_fallback_data(hours: int = 168) -> pd.DataFrame:
    """
    Genera datos sintéticos con distribuciones realistas respetando el contrato exacto.
    """
    np.random.seed(42)

    # Rango temporal seguro para s390x en UTC
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = [
        (end_dt - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(hours)
    ]
    timestamps.reverse()
    n = len(timestamps)

    demand = np.random.normal(28000, 3000, n)
    gen_nonsync = np.random.uniform(6000, 19000, n)
    gen_sync = np.random.uniform(10000, 20000, n)
    gen_total = gen_nonsync + gen_sync
    nonsync_share = np.clip(gen_nonsync / gen_total, 0.0, 1.0)
    interconnection = np.random.normal(-500, 300, n)
    frequency = np.random.normal(50.0, 0.03, n)
    rocof = np.random.normal(0.0, 0.005, n)

    label_event = (
        (nonsync_share > 0.52) | (np.abs(frequency - 50.0) > 0.07)
    ).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "demand_mw": np.round(demand, 2),
            "gen_total_mw": np.round(gen_total, 2),
            "gen_nonsync_mw": np.round(gen_nonsync, 2),
            "gen_sync_mw": np.round(gen_sync, 2),
            "nonsync_share": np.round(nonsync_share, 4),
            "interconnection_balance_mw": np.round(interconnection, 2),
            "frequency_hz": np.round(frequency, 3),
            "rocof_hz_s": np.round(rocof, 4),
            "label_event": label_event,
        }
    )

    return df


def build_and_save_dataset():
    """
    Ejecuta el pipeline de ingesta y guarda en data/features.parquet.
    """
    print("📡 Iniciando ingesta de datos para GridZ Shield...")

    now_utc = datetime.now(timezone.utc)
    start_str = (now_utc - timedelta(days=6)).strftime("%Y-%m-%d")
    end_str = now_utc.strftime("%Y-%m-%d")

    df = fetch_redata_structure(start_str, end_str)

    if df is None or df.empty:
        print("🔄 Generando dataset sintético de respaldo (Fallback)...")
        df = generate_fallback_data()
    else:
        # Formatear timestamp a ISO string antes de guardar
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if "nonsync_share" not in df.columns:
        df["nonsync_share"] = (df["gen_nonsync_mw"] / df["gen_total_mw"]).clip(
            0.0, 1.0
        )

    required_columns = [
        "timestamp",
        "demand_mw",
        "gen_total_mw",
        "gen_nonsync_mw",
        "gen_sync_mw",
        "nonsync_share",
        "interconnection_balance_mw",
        "frequency_hz",
        "rocof_hz_s",
        "label_event",
    ]

    df_final = df[required_columns].copy()
    df_final = df_final.sort_values("timestamp").reset_index(drop=True)

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    df_final.to_parquet(DATASET_PATH, index=False)

    print(
        f"✅ Archivo Parquet generado con éxito en '{DATASET_PATH}' ({len(df_final)} filas)."
    )


if __name__ == "__main__":
    build_and_save_dataset()
    