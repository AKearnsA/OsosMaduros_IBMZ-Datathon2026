"""
Modulo de Ingesta de Datos y Conectores para GridZ Shield.
Conecta con la API publica de Red Electrica (REData) y genera data/features.parquet.
Optimizado para arquitectura IBM LinuxONE (s390x).

Uso (desde la raiz del repo):
    python3 data/ingest_apis.py
"""

import os
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

DATASET_PATH = "data/features.parquet"

# fastparquet: instala en s390x sin compilar Arrow (ver requirements.txt).
PARQUET_ENGINE = "fastparquet"

REDATA_BASE_URL = "https://apidatos.ree.es/es/datos"
GENERATION_ENDPOINT = "generacion/estructura-generacion"
DEMAND_ENDPOINT = "demanda/evolucion"
# Saldo programado (el endpoint de flujos fisicos no esta disponible).
EXCHANGES_ENDPOINT = "intercambios/todas-fronteras-programados"
# Sistema peninsular: Baleares y Canarias son sistemas aislados con su propia frecuencia.
REDATA_GEO_PARAMS = "&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741"

# Tecnologias renovables variables (eolica y solar), tratadas como no sincronas.
# Decision del equipo: la solar termica se agrupa con la solar aunque use turbina de vapor.
NONSYNC_TECHS = {"eolica", "solar fotovoltaica", "solar termica"}
# Subtotal que ya devuelve la API: sumarlo duplicaria la generacion.
EXCLUDED_TECHS = {"generacion total"}

# Constante de inercia tipica H [s] por tecnologia con masa rotativa.
# Fuente: OECD-NEA (2020), "Operation and control of power systems with reduced
# synchronous inertia", diapositiva 25 (Continental Europe, EUCO30 & TYNDP2018).
# Eolica y FV no aportan inercia (H = 0). La solar termica cuenta como no sincrona
# en nonsync_share, pero su turbina de vapor si aporta inercia fisica.
INERTIA_H_SECONDS = {
    "eolica": 0.0,
    "solar fotovoltaica": 0.0,
    "solar termica": 3.0,  # Solar-thermal
    "nuclear": 6.0,  # Nuclear
    "carbon": 4.1,  # Hard coal
    "ciclo combinado": 4.3,  # Gas
    "fuel + gas": 4.3,  # Oil (THN)
    "turbina de vapor": 4.3,  # Oil (THN)
    "turbina de gas": 4.3,  # Gas
    "motores diesel": 4.3,  # Oil (THN)
    "hidraulica": 4.0,  # Hydro-dam (mayoritaria en la peninsula)
    "otras renovables": 3.3,  # Biofuels
    "residuos renovables": 3.3,  # Biofuels
    "cogeneracion": 3.7,  # Other non-RES
    "residuos no renovables": 3.7,  # Other non-RES
}
# Tecnologia no listada arriba.
DEFAULT_H_SECONDS = 3.7
# H medio usado solo en el fallback sintetico (sin desglose por tecnologia).
FALLBACK_H_SECONDS = 4.3

NOMINAL_FREQUENCY_HZ = 50.0
# Perdida de referencia para el RoCoF estimado: un grupo nuclear grande (~1 GW).
REFERENCE_LOSS_MW = 1000.0

# Dias (hora peninsular) con un evento de inestabilidad observado y documentado.
OBSERVED_EVENT_DATES = {"2025-04-28"}  # Apagon iberico

# Regla proxy (NO es un evento observado): dia con cuota no sincrona alta.
NONSYNC_PROXY_THRESHOLD = 0.55

REQUIRED_COLUMNS = [
    "timestamp",
    "demand_mw",
    "gen_total_mw",
    "gen_nonsync_mw",
    "gen_sync_mw",
    "nonsync_share",
    "interconnection_balance_mw",
    "label_event",
    "label_proxy",
    "inertia_est_gws",
    "rocof_est_hz_s",
]


def normalize_tech_name(name: str) -> str:
    """Minusculas y sin tildes, para no depender de la codificacion del sistema."""
    decomposed = unicodedata.normalize("NFKD", name)
    return (
        "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()
    )


def add_labels(df: pd.DataFrame, local_dates: pd.Series) -> pd.DataFrame:
    """label_event solo para eventos observados; la regla proxy va en label_proxy."""
    is_event = local_dates.isin(OBSERVED_EVENT_DATES)
    df["label_event"] = pd.array(np.where(is_event, 1, pd.NA), dtype="Int8")
    df["label_proxy"] = (df["nonsync_share"] > NONSYNC_PROXY_THRESHOLD).astype("int8")
    return df


def add_inertia_estimates(
    df: pd.DataFrame, kinetic_energy_mws: pd.Series
) -> pd.DataFrame:
    """
    Estimaciones fisicas (no medidas) a partir del mix de generacion. Sustituyen
    a la frecuencia medida, que no publica ninguna fuente abierta sin token:
    - inertia_est_gws: energia cinetica E = sum(H_i * P_i), en GW*s. Usa la
      potencia media generada como aproximacion de la potencia nominal en linea,
      por lo que tiende a infraestimar la inercia (criterio conservador).
    - rocof_est_hz_s: RoCoF inicial ante la perdida de REFERENCE_LOSS_MW con el
      sistema peninsular aislado: RoCoF = dP * f0 / (2 * E).
    """
    kinetic_energy_mws = kinetic_energy_mws.where(kinetic_energy_mws > 0)
    df["inertia_est_gws"] = kinetic_energy_mws / 1000.0
    df["rocof_est_hz_s"] = (
        REFERENCE_LOSS_MW * NOMINAL_FREQUENCY_HZ / (2.0 * kinetic_energy_mws)
    )
    return df


def month_ranges(start_year: int, end_year: int) -> list[tuple[str, str]]:
    """Bloques mensuales (inicio, fin) para no superar los limites de la API."""
    ranges = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = f"{year}-{month:02d}-01T00:00"
            if month == 12:
                end_date = f"{year}-12-31T23:59"
            else:
                next_month_first = datetime(year, month + 1, 1)
                last_day = next_month_first - timedelta(days=1)
                end_date = f"{last_day.strftime('%Y-%m-%d')}T23:59"
            ranges.append((start_date, end_date))
    return ranges


def fetch_redata_block(endpoint: str, start_date: str, end_date: str) -> list[dict]:
    """Descarga un bloque diario peninsular de REData; devuelve 'included' o []."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    url = (
        f"{REDATA_BASE_URL}/{endpoint}?start_date={start_date}&end_date={end_date}"
        f"&time_trunc=day{REDATA_GEO_PARAMS}"
    )

    try:
        response = requests.get(url, headers=headers, timeout=12)
        if response.ok:
            return response.json().get("included", [])
        print(
            f"[!] Error en {endpoint} bloque {start_date[:7]} "
            f"[HTTP {response.status_code}]"
        )
    except Exception as e:
        print(f"[!] Excepcion consultando {endpoint} bloque {start_date[:7]}: {e}")
    return []


def parse_generation(included: list[dict]) -> list[dict]:
    """Energia diaria (MWh) por tecnologia."""
    records = []
    for item in included:
        attributes = item.get("attributes", {})
        tech_title = attributes.get("title", item.get("type", "Desconocido"))
        for entry in attributes.get("values", []):
            records.append(
                {
                    "datetime_local": entry.get("datetime"),
                    "tech": normalize_tech_name(tech_title),
                    "value_mwh": entry.get("value", 0.0),
                }
            )
    return records


def parse_demand(included: list[dict]) -> list[dict]:
    """Demanda diaria (MWh)."""
    return [
        {"datetime_local": entry.get("datetime"), "value_mwh": entry.get("value")}
        for item in included
        for entry in item.get("attributes", {}).get("values", [])
    ]


def parse_exchanges(included: list[dict]) -> list[dict]:
    """Saldo diario (MWh) por frontera: + importacion, - exportacion."""
    records = []
    for country in included:
        for series in country.get("attributes", {}).get("content", []):
            if series.get("type") != "saldo":
                continue
            for entry in series.get("attributes", {}).get("values", []):
                records.append(
                    {
                        "datetime_local": entry.get("datetime"),
                        "value_mwh": entry.get("value"),
                    }
                )
    return records


def daily_total(records: list[dict], index: pd.Index) -> pd.Series:
    """Suma por dia y alinea con los dias de generacion (NaN si falta el dato)."""
    if not records:
        return pd.Series(np.nan, index=index)
    df = pd.DataFrame(records)
    return df.groupby("datetime_local")["value_mwh"].sum().reindex(index)


def fetch_redata_structure(
    start_year: int = 2021, end_year: int = 2025
) -> Optional[pd.DataFrame]:
    """
    Obtiene de REData (sistema peninsular, resolucion diaria) la estructura de
    generacion, la demanda y el saldo de intercambios internacionales,
    realizando peticiones por bloques mensuales.
    """
    gen_records, demand_records, exchange_records = [], [], []

    for start_date, end_date in month_ranges(start_year, end_year):
        gen_records += parse_generation(
            fetch_redata_block(GENERATION_ENDPOINT, start_date, end_date)
        )
        demand_records += parse_demand(
            fetch_redata_block(DEMAND_ENDPOINT, start_date, end_date)
        )
        exchange_records += parse_exchanges(
            fetch_redata_block(EXCHANGES_ENDPOINT, start_date, end_date)
        )

    if not gen_records:
        print("[!] No se pudieron obtener registros de REData.")
        return None

    df_raw = pd.DataFrame(gen_records)
    df_raw = df_raw[~df_raw["tech"].isin(EXCLUDED_TECHS)]

    # Energia diaria (MWh) por tecnologia; una fila por dia.
    df_pivoted = df_raw.pivot_table(
        index="datetime_local",
        columns="tech",
        values="value_mwh",
        aggfunc="sum",
    ).fillna(0.0)

    nonsync_cols = [c for c in df_pivoted.columns if c in NONSYNC_TECHS]
    sync_cols = [c for c in df_pivoted.columns if c not in NONSYNC_TECHS]

    # MWh/dia -> MW medios del dia (dias de cambio de hora tienen 23 o 25 h).
    hours_per_day = 24.0
    df_features = pd.DataFrame(index=df_pivoted.index)
    df_features["gen_nonsync_mw"] = df_pivoted[nonsync_cols].sum(axis=1) / hours_per_day
    df_features["gen_sync_mw"] = df_pivoted[sync_cols].sum(axis=1) / hours_per_day
    df_features["gen_total_mw"] = (
        df_features["gen_nonsync_mw"] + df_features["gen_sync_mw"]
    )
    df_features["nonsync_share"] = (
        df_features["gen_nonsync_mw"] / df_features["gen_total_mw"]
    ).clip(0.0, 1.0)

    df_features["demand_mw"] = (
        daily_total(demand_records, df_features.index) / hours_per_day
    )
    df_features["interconnection_balance_mw"] = (
        daily_total(exchange_records, df_features.index) / hours_per_day
    )

    unknown_techs = [c for c in df_pivoted.columns if c not in INERTIA_H_SECONDS]
    if unknown_techs:
        print(
            f"[!] Sin H especifico, se usa {DEFAULT_H_SECONDS} s para: {unknown_techs}"
        )
    h_by_tech = pd.Series(
        {c: INERTIA_H_SECONDS.get(c, DEFAULT_H_SECONDS) for c in df_pivoted.columns}
    )
    kinetic_energy_mws = df_pivoted.mul(h_by_tech, axis=1).sum(axis=1) / hours_per_day
    df_features = add_inertia_estimates(df_features, kinetic_energy_mws)

    df_features = df_features.reset_index()
    local_dates = df_features["datetime_local"].str[:10]
    df_features["timestamp"] = pd.to_datetime(df_features["datetime_local"], utc=True)
    df_features = add_labels(df_features, local_dates)

    print(
        f"[OK] Se obtuvieron {len(df_features)} dias reales de REData (sistema peninsular); "
        f"demanda en {df_features['demand_mw'].notna().sum()}, "
        f"intercambios en {df_features['interconnection_balance_mw'].notna().sum()}."
    )
    return df_features


def generate_fallback_data(hours: int = 168) -> pd.DataFrame:
    """
    Genera datos SINTETICOS con distribuciones realistas.
    Solo para que el resto del pipeline pueda probarse sin conexion.
    """
    print("[!] ATENCION: REData no disponible. Se generan datos SINTETICOS (Fallback).")
    np.random.seed(42)

    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = [end_dt - timedelta(hours=i) for i in range(hours)]
    timestamps.reverse()
    n = len(timestamps)

    demand = np.random.normal(28000, 3000, n)
    gen_nonsync = np.random.normal(12000, 2500, n)
    gen_sync = np.random.normal(17000, 2000, n)
    gen_total = gen_nonsync + gen_sync
    nonsync_share = np.clip(gen_nonsync / gen_total, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "demand_mw": np.round(demand, 2),
            "gen_total_mw": np.round(gen_total, 2),
            "gen_nonsync_mw": np.round(gen_nonsync, 2),
            "gen_sync_mw": np.round(gen_sync, 2),
            "nonsync_share": np.round(nonsync_share, 4),
            "interconnection_balance_mw": np.round(np.random.normal(-500, 300, n), 2),
        }
    )
    df = add_inertia_estimates(df, df["gen_sync_mw"] * FALLBACK_H_SECONDS)
    # Ningun evento observado en datos sinteticos.
    return add_labels(df, pd.Series([""] * n))


def verify_roundtrip(df_written: pd.DataFrame, path: str) -> None:
    """Relee el Parquet y comprueba que los valores coinciden (big-endian en s390x)."""
    df_read = pd.read_parquet(path, engine=PARQUET_ENGINE)
    pd.testing.assert_frame_equal(
        df_written.reset_index(drop=True),
        df_read.reset_index(drop=True),
        check_dtype=False,
    )


def build_and_save_dataset() -> None:
    """
    Ejecuta la ingesta desde REData. Si falla, activa el fallback sintetico
    y guarda el archivo Parquet de forma segura para s390x.
    """
    print("[*] Iniciando ingesta de datos para GridZ Shield...")
    df = fetch_redata_structure()

    if df is None or df.empty:
        df = generate_fallback_data()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    df_final = df[REQUIRED_COLUMNS].copy()
    df_final = df_final.sort_values("timestamp").reset_index(drop=True)

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    df_final.to_parquet(DATASET_PATH, index=False, engine=PARQUET_ENGINE)
    verify_roundtrip(df_final, DATASET_PATH)

    print(
        f"[OK] Parquet generado y verificado en '{DATASET_PATH}' "
        f"({len(df_final)} filas, label_proxy=1 en {int(df_final['label_proxy'].sum())})."
    )


if __name__ == "__main__":
    build_and_save_dataset()
