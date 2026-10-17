"""
GridZ Shield - Módulo 2: Entrenamiento Predictivo (Regresión t+1 + Anomalías)
Ubicación: model/train.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "features.parquet"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "grid_model.joblib"


def load_and_prepare_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado en {file_path}")

    df = pd.read_parquet(file_path).copy()
    
    # 1. Asegurar orden temporal por timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Filtrar el día del apagón real si existe label_event == 1 para evitar valores distorsionados
    if "label_event" in df.columns:
        df = df[df["label_event"] == 0].copy()

    # 3. Ingeniería de Features Temporales y Lags
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Lags (Ayer)
    df["rocof_lag1"] = df["rocof_est_hz_s"].shift(1)
    df["nonsync_share_lag1"] = df["nonsync_share"].shift(1)
    
    # Medias móviles (7 días)
    df["rocof_roll7_mean"] = df["rocof_est_hz_s"].rolling(window=7).mean()
    df["nonsync_roll7_mean"] = df["nonsync_share"].rolling(window=7).mean()

    # 4. Target de Regresión: el RoCoF del día siguiente (t+1)
    df["target_rocof_next_day"] = df["rocof_est_hz_s"].shift(-1)

    # Eliminar filas con NaN generados por shifts/rolling
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"[+] Dataset preparado: {len(df_clean)} filas limpias.")
    return df_clean


def train_and_evaluate(df: pd.DataFrame):
    # Variables de entrada para el modelo regressivo
    feature_cols = [
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
        "day_of_week",
        "month",
    ]

    # División por FECHAS (Sin Shuffle): Entrenar 2021-2024, Probar 2025
    train_mask = df["timestamp"].dt.year < 2025
    test_mask = df["timestamp"].dt.year == 2025

    df_train = df[train_mask]
    df_test = df[test_mask]

    X_train, y_train = df_train[feature_cols], df_train["target_rocof_next_day"]
    X_test, y_test = df_test[feature_cols], df_test["target_rocof_next_day"]

    print(f"[+] División Temporal -> Train (2021-2024): {len(X_train)} | Test (2025): {len(X_test)}")

    # --- Pieza A: Predicción del RoCoF de mañana (RandomForestRegressor) ---
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae_model = mean_absolute_error(y_test, y_pred)
    rmse_model = np.sqrt(mean_squared_error(y_test, y_pred))

    # Baseline iluso: "Mañana el RoCoF será igual al de hoy"
    y_baseline = df_test["rocof_est_hz_s"]
    mae_baseline = mean_absolute_error(y_test, y_baseline)

    print("\n" + "=" * 55)
    print("      EVALUACIÓN DE REGRESIÓN (Predicción t+1)")
    print("=" * 55)
    print(f"  MAE Modelo IA  : {mae_model:.6f} Hz/s")
    print(f"  MAE Baseline   : {mae_baseline:.6f} Hz/s")
    
    if mae_model < mae_baseline:
        improvement = ((mae_baseline - mae_model) / mae_baseline) * 100
        print(f"  [✓] La IA mejora al baseline en un {improvement:.2f}%")
    else:
        print("  [!] Advertencia: La IA no supera el baseline iluso.")
    print("=" * 55)

    # --- Pieza B: Detector de Anomalías (IsolationForest) ---
    anomaly_cols = [
        "nonsync_share",
        "inertia_est_gws",
        "demand_mw",
        "interconnection_balance_mw",
        "gen_sync_mw",
    ]
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(df_train[anomaly_cols])

    # --- Pieza C: Cálculo de Percentiles de Referencia ---
    train_rocof_values = y_train.values
    p75 = np.percentile(train_rocof_values, 75)
    p95 = np.percentile(train_rocof_values, 95)

    print(f"\n[+] Umbrales de Riesgo Basados en Percentiles (Train 2021-2024):")
    print(f"    - Percentil 75 (Watch)   : {p75:.4f} Hz/s")
    print(f"    - Percentil 95 (Critical): {p95:.4f} Hz/s")

    # Guardar todos los artefactos necesarios para la API
    artifact = {
        "regressor_model": reg,
        "anomaly_model": iso_forest,
        "feature_names": feature_cols,
        "anomaly_feature_names": anomaly_cols,
        "percentile_75": p75,
        "percentile_95": p95,
        "train_rocof_distribution": train_rocof_values,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[✓] Artefactos exportados exitosamente a: {MODEL_PATH}")


def main():
    print("=== GridZ Shield: Entrenamiento Predictivo & Anomalías ===")
    df = load_and_prepare_dataset(DATA_PATH)
    train_and_evaluate(df)
    print("=== Proceso finalizado con éxito ===")


if __name__ == "__main__":
    main()
    