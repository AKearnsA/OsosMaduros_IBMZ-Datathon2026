import os
import joblib
import pandas as pd
from typing import Tuple, List

MODEL_PATH = os.getenv("GRID_MODEL_PATH", "model/grid_model.joblib")

class GridPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el artefacto en {model_path}.")
        
        loaded = joblib.load(model_path)
        
        # Si el objeto guardado es un diccionario con artefactos
        if isinstance(loaded, dict):
            # Intenta obtener el regresor con varias claves comunes o el valor directo
            self.model = loaded.get("regressor") or loaded.get("model") or loaded.get("grid_model")
            self.p75 = loaded.get("p75", 0.3401)
            self.p95 = loaded.get("p95", 0.4019)
            self.feature_names = loaded.get("feature_names", [
                "nonsync_share", "interconnection_balance_mw", "demand_mw",
                "inertia_est_gws", "rocof_lag_1d", "rocof_ma_7d", "month", "dayofweek"
            ])
        else:
            # Si se guardó el modelo directamente en el joblib
            self.model = loaded
            self.p75 = 0.3401
            self.p95 = 0.4019
            self.feature_names = [
                "nonsync_share", "interconnection_balance_mw", "demand_mw",
                "inertia_est_gws", "rocof_lag_1d", "rocof_ma_7d", "month", "dayofweek"
            ]

        if self.model is None:
            raise ValueError(f"No se pudo extraer el modelo del archivo {model_path}.")

    def predict(self, data: dict) -> Tuple[float, str, List[str]]:
        gen_total = data["gen_total_mw"] if data["gen_total_mw"] > 0 else 1.0
        nonsync_share = data["gen_nonsync_mw"] / gen_total
        inertia_est_gws = (data["gen_sync_mw"] * 4.5) / 1000.0
        dt = pd.to_datetime(data["timestamp"])

        features_dict = {
            "demand_mw": [data["demand_mw"]],
            "gen_total_mw": [data["gen_total_mw"]],
            "gen_nonsync_mw": [data["gen_nonsync_mw"]],
            "gen_sync_mw": [data["gen_sync_mw"]],
            "nonsync_share": [nonsync_share],
            "interconnection_balance_mw": [data["interconnection_balance_mw"]],
            "inertia_est_gws": [inertia_est_gws],
            "rocof_lag_1d": [data.get("rocof_lag_1d", 0.0)],
            "rocof_ma_7d": [data.get("rocof_ma_7d", 0.0)],
            "month": [dt.month],
            "dayofweek": [dt.dayofweek]
        }

        # Filtrar solo las columnas que espera el modelo
        df_input = pd.DataFrame(features_dict)
        cols_to_use = [col for col in self.feature_names if col in df_input.columns]
        X = df_input[cols_to_use] if cols_to_use else df_input

        pred_rocof = float(self.model.predict(X)[0])

        # Mapeo según percentiles históricos
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

        top_factors = []
        if nonsync_share > 0.45:
            top_factors.append("nonsync_share")
        if abs(data["interconnection_balance_mw"]) > 2000:
            top_factors.append("interconnection_balance_mw")
        if inertia_est_gws < 15.0:
            top_factors.append("inertia_est_gws")

        if not top_factors:
            top_factors = ["nonsync_share", "interconnection_balance_mw"]

        return risk_score, level, top_factors[:2]
        