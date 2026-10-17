"""
Genera un features.parquet SINTETICO con el mismo esquema que data/ingest_apis.py,
para probar el resto del pipeline sin conexion.

Uso (desde la raiz del repo):
    python3 data/create_mock_features.py
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ingest_apis import (
    FALLBACK_H_SECONDS,
    PARQUET_ENGINE,
    REQUIRED_COLUMNS,
    add_inertia_estimates,
    add_labels,
    verify_roundtrip,
)


def generate_mock_data(output_path: str = "data/features.parquet") -> pd.DataFrame:
    """
    Genera un dataset sintetico en formato Parquet que cumple con el
    Contrato de Datos obligatorio para GridZ Shield.
    """
    # Rango temporal de 24h a intervalos de 5 minutos (288 registros)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=1)
    timestamps = pd.date_range(start=start_time, end=end_time, freq="5min", tz="UTC")
    n_samples = len(timestamps)

    # Variables de generacion y demanda (en MW)
    demand_mw = np.random.uniform(22000, 34000, size=n_samples)
    gen_nonsync_mw = np.random.uniform(6000, 19000, size=n_samples)  # Eolica + Solar
    gen_sync_mw = (
        demand_mw - gen_nonsync_mw + np.random.uniform(-400, 400, size=n_samples)
    )
    gen_total_mw = gen_nonsync_mw + gen_sync_mw

    # nonsync_share: proporcion de generacion no sincrona (0.0 a 1.0)
    nonsync_share = np.clip(gen_nonsync_mw / gen_total_mw, 0.0, 1.0)

    # Balance de interconexion (+ importacion / - exportacion)
    interconnection_balance_mw = np.random.uniform(-2500, 2500, size=n_samples)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "demand_mw": demand_mw.astype(float),
            "gen_total_mw": gen_total_mw.astype(float),
            "gen_nonsync_mw": gen_nonsync_mw.astype(float),
            "gen_sync_mw": gen_sync_mw.astype(float),
            "nonsync_share": nonsync_share.astype(float),
            "interconnection_balance_mw": interconnection_balance_mw.astype(float),
        }
    )
    df = add_inertia_estimates(df, df["gen_sync_mw"] * FALLBACK_H_SECONDS)
    # Ningun evento observado en datos sinteticos; label_proxy segun la regla comun.
    df = add_labels(df, pd.Series([""] * n_samples))
    df = df[REQUIRED_COLUMNS]

    df.to_parquet(output_path, index=False, engine=PARQUET_ENGINE)
    verify_roundtrip(df, output_path)
    print(f"[OK] Dataset SINTETICO generado: {output_path}")
    print(f"[+] Registros: {len(df)} | Columnas: {list(df.columns)}")
    return df


if __name__ == "__main__":
    generate_mock_data()
