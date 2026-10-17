import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

def generate_mock_data(output_path="data/features.parquet"):
    """
    Genera un dataset sintético en formato Parquet que cumple con el 
    Contrato de Datos obligatorio para GridZ Shield.
    """
    # Rango temporal de 24h a intervalos de 5 minutos (288 registros)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=1)
    timestamps = pd.date_range(start=start_time, end=end_time, freq="5min", tz="UTC")
    n_samples = len(timestamps)

    # Variables de generación y demanda (en MW)
    demand_mw = np.random.uniform(22000, 34000, size=n_samples)
    gen_nonsync_mw = np.random.uniform(6000, 19000, size=n_samples)  # Eólica + Solar PV
    gen_sync_mw = demand_mw - gen_nonsync_mw + np.random.uniform(-400, 400, size=n_samples)
    gen_total_mw = gen_nonsync_mw + gen_sync_mw

    # nonsync_share: proporción de generación no sincrónica (0.0 a 1.0)
    nonsync_share = np.clip(gen_nonsync_mw / gen_total_mw, 0.0, 1.0)

    # Balance de interconexión (+ importación / - exportación)
    interconnection_balance_mw = np.random.uniform(-2500, 2500, size=n_samples)

    # Frecuencia (Hz) y RoCoF (Hz/s)
    frequency_hz = np.random.normal(50.0, 0.04, size=n_samples)
    time_diff_sec = 300.0  # 5 min = 300 segundos
    rocof_hz_s = np.insert(np.diff(frequency_hz) / time_diff_sec, 0, 0.0)

    # Proxy de evento de inestabilidad (1 = Inestable, 0 = Estable)
    label_event = ((nonsync_share > 0.55) | (np.abs(frequency_hz - 50.0) > 0.10)).astype(int)

    # Construir DataFrame con el esquema exacto
    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": demand_mw.astype(float),
        "gen_total_mw": gen_total_mw.astype(float),
        "gen_nonsync_mw": gen_nonsync_mw.astype(float),
        "gen_sync_mw": gen_sync_mw.astype(float),
        "nonsync_share": nonsync_share.astype(float),
        "interconnection_balance_mw": interconnection_balance_mw.astype(float),
        "frequency_hz": frequency_hz.astype(float),
        "rocof_hz_s": rocof_hz_s.astype(float),
        "label_event": label_event.astype(int)
    })

    # Exportar usando fastparquet
    df.to_parquet(output_path, engine="fastparquet")
    print(f"[✓] Dataset generado con éxito: {output_path}")
    print(f"[+] Registros: {len(df)} | Columnas: {list(df.columns)}")
    return df

if __name__ == "__main__":
    generate_mock_data()
