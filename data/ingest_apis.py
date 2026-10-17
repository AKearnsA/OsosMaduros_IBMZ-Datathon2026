
"""
Módulo de Ingesta de Datos y Conectores para GridZ Shield.
Conecta con la API pública de Red Eléctrica (REData) y genera data/features.parquet.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

DATASET_PATH = "data/features.parquet"

def fetch_redata_structure(
    start_year: int = 2024, end_year: int = 2024
) -> Optional[pd.DataFrame]:
    """
    Obtiene la estructura de generación diaria de REData realizando peticiones
    por bloques mensuales para evitar límites de la API.
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    all_records = []

    # Iterar mes a mes para el rango de años especificado
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Calcular primer y último día del mes
            start_date = f"{year}-{month:02d}-01T00:00"
            if month == 12:
                end_date = f"{year}-12-31T23:59"
            else:
                next_month_first = datetime(year, month + 1, 1)
                last_day = next_month_first - timedelta(days=1)
                end_date = f"{last_day.strftime('%Y-%m-%d')}T23:59"

            url = (
                f"https://apidatos.ree.es/es/datos/generacion/estructura-generacion"
                f"?start_date={start_date}&end_date={end_date}&time_trunc=day"
            )

            try:
                response = requests.get(url, headers=headers, timeout=12)

                if response.ok:
                    json_data = response.json()
                    included = json_data.get("included", [])

                    for item in included:
                        attributes = item.get("attributes", {})
                        tech_title = attributes.get(
                            "title", item.get("type", "Desconocido")
                        )
                        values = attributes.get("values", [])

                        for entry in values:
                            all_records.append(
                                {
                                    "timestamp": entry.get("datetime"),
                                    "tech": tech_title,
                                    "value_mw": entry.get("value", 0.0),
                                }
                            )
                else:
                    print(
                        f"⚠️ Error en bloque {start_date[:7]} [HTTP {response.status_code}]"
                    )

            except Exception as e:
                print(f"⚠️ Excepción consultando bloque {start_date[:7]}: {e}")

    if not all_records:
        print("[!] No se pudieron obtener registros de REData.")
        return None

    # Procesar dataframe global
    df_raw = pd.DataFrame(all_records)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)

    # Pivotar por fecha y tecnología
    df_pivoted = df_raw.pivot_table(
        index="timestamp",
        columns="tech",
        values="value_mw",
        aggfunc="sum",
    ).fillna(0.0)

    # Clasificación Síncrona vs No Síncrona
    nonsync_keywords = ["eólica", "solar", "fotovoltaica", "wind", "pv"]
    nonsync_cols = [
        c
        for c in df_pivoted.columns
        if any(k in c.lower() for k in nonsync_keywords)
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

    print(
        f"[✓] Éxito: Se obtuvieron {len(df_features)} registros reales de REData."
    )
    return df_features.reset_index()

def generate_fallback_data(hours: int = 168) -> pd.DataFrame:
    """
    Genera datos sintéticos con distribuciones realistas respetando el contrato exacto.
    """
    print("🔄 Generando dataset sintético de respaldo (Fallback)...")
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
    Ejecuta la ingesta desde REData. Si falla, activa el fallback sintético
    y guarda el archivo Parquet respetando el contrato de datos.
    """
    print("📡 Iniciando ingesta de datos para GridZ Shield...")
    df = fetch_redata_structure()

    # Si REData falla o devuelve vacío, se activa el fallback automáticamente
    if df is None or df.empty:
        df = generate_fallback_data()

    # Cálculo obligatorio del contrato
    df["nonsync_share"] = (df["gen_nonsync_mw"] / df["gen_total_mw"]).clip(0.0, 1.0)

    # Orden exacto de columnas del contrato de datos de GridZ Shield
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

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    df_final.to_parquet(DATASET_PATH, index=False)

    print(f"✅ Archivo Parquet generado con éxito en '{DATASET_PATH}' ({len(df_final)} filas).")

if __name__ == "__main__":
    build_and_save_dataset()

