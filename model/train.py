"""
GridZ Shield - Módulo 2: Entrenamiento del Modelo ML
Ubicación: model/train.py

Entrena un modelo RandomForestClassifier para predecir inestabilidades
en la red eléctrica (label_event) a partir de variables físicas.
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

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "features.parquet"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "grid_model.joblib"

# Contrato de Atributos (Features) según las especificaciones de GridZ Shield
FEATURE_COLS = [
    "demand_mw",
    "gen_total_mw",
    "gen_nonsync_mw",
    "gen_sync_mw",
    "nonsync_share",
    "interconnection_balance_mw",
    "frequency_hz",
    "rocof_hz_s",
]

TARGET_COL = "label_event"


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Carga la tabla de datos procesada desde Parquet."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en: {file_path}.\n"
            "Ejecuta 'python data/ingest_apis.py' primero."
        )

    df = pd.read_parquet(file_path)
    print(f"[+] Dataset cargado exitosamente: {len(df)} filas.")
    return df


def preprocess_data(df: pd.DataFrame):
    """Limpia imputaciones pendientes y divide en Train / Test."""
    # Eliminar filas con etiquetas nulas si existieran
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    # Imputación ligera para valores nulos opcionales en frecuencia o RoCoF
    for col in FEATURE_COLS:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL].astype(int)

    # Estratificación según distribución de clases
    stratify_target = y if len(y.value_counts()) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_target
    )

    print(f"[+] Datos divididos -> Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
    print(f"[+] Distribución de eventos en Train: {dict(y_train.value_counts())}")

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Entrena el clasificador RandomForest con balanceo de peso por clase."""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo e imprime métricas clave."""
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print("      EVALUACIÓN DEL MODELO - GRIDZ SHIELD")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    if len(clf.classes_) > 1 and len(np.unique(y_test)) > 1:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC   : {auc:.4f}")

    print("\nReporte Detallado de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=" * 55)


def save_artifact(clf: RandomForestClassifier, output_path: Path):
    """Guarda el modelo entrenado y la lista de atributos requerida."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": clf,
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
    }

    joblib.dump(artifact, output_path)
    print(f"\n[✓] Artefacto guardado exitosamente en: {output_path}")


def main():
    print("=== GridZ Shield: Inicio de Entrenamiento del Modelo ===")
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\n[+] Entrenando RandomForestClassifier...")
    clf = train_model(X_train, y_train)

    evaluate_model(clf, X_test, y_test)
    save_artifact(clf, MODEL_PATH)
    print("=== Proceso finalizado correctamente ===")


if __name__ == "__main__":
    main()
    