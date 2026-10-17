"""
Módulo de Ingesta de Datos y Conectores para GridZ Shield.
Conecta con APIs de Red Eléctrica (REData / ESIOS) y genera data/features.parquet.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

DATASET_PATH = "data/features.parquet"
ESIOS_TOKEN = os.getenv("ESIOS_API_TOKEN", None)


def fetch_redata_structure(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Obtiene la estructura de generación de la API pública de REData (Red Eléctrica).
    Endpoint público sin autenticación obligatoria para datos aggregados.
    """
    url = (
        f"https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
        f"?start_date={start_date}T00:00&end_date={end_date}T23:59&time_trunc=hour"
    )
    headers = {"Accept": "application/json"}

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

        # Pivotar por tecnología
        df_pivoted = df_raw.pivot_table(
            index="timestamp", columns="tech", values="value_mw", aggfunc="sum"
        ).fillna(0.0)

        # Clasificación entre síncrona y no síncrona
        nonsync_cols = [c for c in df_pivoted.columns if any(k in c.lower() for k in ["eólica", "solar", "wind", "pv"])]
        sync_cols = [c for c in df_pivoted.columns if c not in nonsync_cols]

        df_features = pd.DataFrame(index=df_pivoted.index)
        df_features["gen_nonsync_mw"] = df_pivoted[nonsync_cols].sum(axis=1) if nonsync_cols else 0.0
        df_features["gen_sync_mw"] = df_pivoted[sync_cols].sum(axis=1) if sync_cols else 10000.0
        df_features["gen_total_mw"] = df_features["gen_nonsync_mw"] + df_features["gen_sync_mw"]
        df_features["demand_mw"] = df_features["gen_total_mw"] * 0.98  # Estimación base
        df_features["interconnection_balance_mw"] = 0.0  # Balance neutro por defecto
        df_features["frequency_hz"] = 50.0
        df_features["rocof_hz_s"] = 0.0
        df_features["label_event"] = 0

        return df_features.reset_index()

    except Exception as e:
        print(f"⚠️ Error al conectar con REData: {e}")
        return None


def generate_fallback_data(hours: int = 168) -> pd.DataFrame:
    """
    Genera datos sintéticos con distribuciones realistas respetando el contrato exacto.
    """
    end_time = datetime.now(timezone.utc)
    timestamps = [end_time - timedelta(hours=i) for i in range(hours)]
    timestamps.reverse()

    np.random.seed(42)
    n = len(timestamps)

    demand = np.random.normal(28000, 3000, n)
    gen_nonsync = np.random.normal(12000, 2500, n)
    gen_sync = np.random.normal(17000, 2000, n)
    gen_total = gen_nonsync + gen_sync
    interconnection = np.random.normal(-500, 300, n)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "demand_mw": np.round(demand, 2),
            "gen_total_mw": np.round(gen_total, 2),
            "gen_nonsync_mw": np.round(gen_nonsync, 2),
            "gen_sync_mw": np.round(gen_sync, 2),
            "interconnection_balance_mw": np.round(interconnection, 2),
            "frequency_hz": np.round(np.random.normal(50.0, 0.02, n), 3),
            "rocof_hz_s": np.round(np.random.normal(0.0, 0.005, n), 4),
            "label_event": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        }
    )

    return df


def build_and_save_dataset():
    """
    Ejecuta el pipeline de ingesta, aplica trasformaciones y guarda en features.parquet.
    """
    start_str = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    end_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"📡 Obteniendo datos de telemetría ({start_str} a {end_str})...")
    df = fetch_redata_structure(start_str, end_str)

    if df is None or df.empty:
        print("🔄 Usando generador sintético de respaldo (Fallback)...")
        df = generate_fallback_data()

    # Cálculo obligatoria del contrato: nonsync_share
    df["nonsync_share"] = (df["gen_nonsync_mw"] / df["gen_total_mw"]).clip(0.0, 1.0)

    # Orden exacto del contrato de datos de GridZ Shield
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
    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"], utc=True)
    df_final = df_final.sort_values("timestamp").reset_index(drop=True)

    # Crear la carpeta data/ si no existe y guardar parquet
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    df_final.to_parquet(DATASET_PATH, index=False)

    print(f"✅ Archivo Parquet generado con éxito en '{DATASET_PATH}' ({len(df_final)} filas).")


if __name__ == "__main__":
    build_and_save_dataset()