"""
GridZ Shield - Módulo 2: Entrenamiento del Modelo ML (Actualizado para rama fix/data-ingest)
Ubicación: model/train.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "features.parquet"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "grid_model.joblib"

# Atributos actualizados incluyendo estimación de inercia y RoCoF físico
FEATURE_COLS = [
    "demand_mw",
    "gen_total_mw",
    "gen_nonsync_mw",
    "gen_sync_mw",
    "nonsync_share",
    "interconnection_balance_mw",
    "inertia_est_gws",
    "rocof_est_hz_s",
]

OPERATIONAL_THRESHOLD = 0.25  # Umbral de prevención TSO


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo Parquet no encontrado en: {file_path}")

    df = pd.read_parquet(file_path)
    print(f"[+] Dataset cargado correctamente desde {file_path}: {len(df)} filas.")
    return df


def preprocess_data(df: pd.DataFrame):
    # Usar label_proxy si está presente, de lo contrario label_event
    target_col = "label_proxy" if "label_proxy" in df.columns else "label_event"
    
    df_clean = df.dropna(subset=[target_col]).copy()

    # Imputación de seguridad para columnas con nulos
    for col in FEATURE_COLS:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    X = df_clean[FEATURE_COLS]
    y = df_clean[target_col].astype(int)

    stratify_target = y if len(y.value_counts()) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_target
    )

    print(f"[+] Objetivo de clasificación seleccionado: '{target_col}'")
    print(f"[+] División -> Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[+] Eventos en Train: {dict(y_train.value_counts())}")

    return X_train, X_test, y_train, y_test, target_col


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.50):
    y_proba = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) > 1 else np.zeros(len(y_test))
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print(f"  EVALUACIÓN DEL MODELO (Umbral de Riesgo = {threshold:.2f})")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}  <-- Métrica clave para TSO")
    print(f"  F1-Score  : {f1:.4f}")

    if len(clf.classes_) > 1 and len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC   : {auc:.4f}")

    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 55)


def save_artifact(clf: RandomForestClassifier, target_col: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": clf,
        "feature_names": FEATURE_COLS,
        "target_name": target_col,
        "operational_threshold": OPERATIONAL_THRESHOLD,
    }

    joblib.dump(artifact, output_path)
    print(f"\n[✓] Modelo guardado en: {output_path}")


def main():
    print("=== GridZ Shield: Entrenamiento con Datos Físicos Estimados ===")
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test, target_col = preprocess_data(df)

    print("\n[+] Entrenando RandomForestClassifier...")
    clf = train_model(X_train, y_train)

    print("\n--- 1. Evaluación Estándar (Umbral 0.50) ---")
    evaluate_model(clf, X_test, y_test, threshold=0.50)

    print("\n--- 2. Evaluación Operativa Preventiva (Umbral 0.25) ---")
    evaluate_model(clf, X_test, y_test, threshold=OPERATIONAL_THRESHOLD)

    save_artifact(clf, target_col, MODEL_PATH)
    print("=== Proceso finalizado ===")


if __name__ == "__main__":
    main()
    