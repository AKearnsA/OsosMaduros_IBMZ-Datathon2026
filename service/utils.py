import os
from pathlib import Path

import joblib
import pandas as pd
from typing import Tuple, List

# Ruta relativa a este fichero: funciona se arranque uvicorn desde donde se arranque.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "grid_model.joblib"
MODEL_PATH = os.getenv("GRID_MODEL_PATH", str(DEFAULT_MODEL_PATH))

# Umbrales para top_factors: percentiles de los datos de entrenamiento (2021-2024).
INERTIA_LOW_GWS = 73.5  # p25 de inertia_est_gws
NONSYNC_HIGH = 0.464  # p75 de nonsync_share
INTERCONNECTION_HIGH_MW = 2630.0  # p75 de |interconnection_balance_mw|


class GridPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_path}")

        loaded = joblib.load(model_path)

        if not isinstance(loaded, dict):
            raise ValueError(f"Se esperaba un diccionario en {model_path}, pero se obtuvo {type(loaded)}")

        self.model = loaded.get("regressor_model")
        self.p75 = loaded.get("percentile_75", 0.3401)
        self.p95 = loaded.get("percentile_95", 0.4019)
        self.feature_names = loaded.get("feature_names", [])

        if self.model is None:
            raise ValueError(f"No se encontró 'regressor_model' dentro de {model_path}")

    def predict(self, data: dict) -> Tuple[float, str, List[str]]:
        dt = pd.to_datetime(data["timestamp"])

        # Mismas variables y mismo calculo de fecha que en model/train.py
        features = dict(data)
        features["day_of_week"] = dt.dayofweek
        features["month"] = dt.month

        missing = [col for col in self.feature_names if col not in features]
        if missing:
            raise ValueError(f"Faltan variables que el modelo necesita: {missing}")

        X = pd.DataFrame([features])[self.feature_names]
        pred_rocof = float(self.model.predict(X)[0])

        # Clasificación según percentiles
        if pred_rocof < self.p75:
            level = "Stable"
            risk_score = round(min(0.50, max(0.0, (pred_rocof / self.p75) * 0.50)), 2)
        elif pred_rocof < self.p95:
            level = "Watch"
            ratio = (pred_rocof - self.p75) / (self.p95 - self.p75)
            risk_score = round(0.50 + ratio * 0.35, 2)
        else:
            level = "Critical"
            risk_score = round(min(1.0, 0.85 + (pred_rocof - self.p95) * 1.5), 2)

        return risk_score, level, self._top_factors(data, level)

    @staticmethod
    def _top_factors(data: dict, level: str) -> List[str]:
        """
        Ordena los factores por cuanto superan su umbral (>1 = fuera de lo normal).
        En un dia Stable sin ningun factor fuera de lo normal devuelve [] para no
        presentar como riesgo algo que no lo es.
        """
        severity = {
            "inertia_est_gws": INERTIA_LOW_GWS / data["inertia_est_gws"],
            "nonsync_share": data["nonsync_share"] / NONSYNC_HIGH,
            "interconnection_balance_mw": abs(data["interconnection_balance_mw"])
            / INTERCONNECTION_HIGH_MW,
        }
        ranked = sorted(severity, key=severity.get, reverse=True)
        above_threshold = [f for f in ranked if severity[f] > 1.0]

        if above_threshold:
            return above_threshold[:2]
        # Watch/Critical sin ningun factor fuera de umbral: se indica el mas cercano.
        return ranked[:1] if level != "Stable" else []
