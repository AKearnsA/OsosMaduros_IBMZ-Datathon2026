import os
import json
import subprocess
from explain.schemas import ExplanationRequest, ExplanationResponse


class GraniteExplainer:
    def __init__(
        self,
        model_path: str = os.getenv("GRANITE_MODEL_PATH", "models/granite-3.0-3b-instruct.Q4_K_M.gguf"),
        binary_path: str = os.getenv("LLAMA_BINARY_PATH", "./llama.cpp/main")
    ):
        self.model_path = model_path
        self.binary_path = binary_path
        self.has_llm = os.path.exists(self.model_path) and os.path.exists(self.binary_path)

    def _build_prompt(self, data: ExplanationRequest) -> str:
        return f"""<|system|>
Eres el asistente técnico de GridZ Shield en un servidor IBM LinuxONE.
Genera una explicación breve y exactamente 2 acciones correctivas.
Responde ÚNICAMENTE un JSON válido con este formato:
{{"summary": "texto", "recommended_actions": ["accion1", "accion2"], "technical_details": "texto"}}
<|user|>
Evento: Riesgo {data.risk_score:.2f} ({data.level}). Factores: {', '.join(data.top_factors)}.
<|assistant|>
"""

    def _rule_based_fallback(self, data: ExplanationRequest) -> ExplanationResponse:
        """Fallback determinista si no está el binario o falla el SLM."""
        actions_map = {
            "nonsync_share": "Monitorear inercia y preparar reservas síncronas.",
            "interconnection_balance_mw": "Ajustar flujos en el enlace de interconexión.",
            "inertia_est_gws": "Activar inercia sintética o potencia síncrona."
        }
        actions = [actions_map[f] for f in data.top_factors if f in actions_map]
        if not actions:
            actions = ["Mantener supervisión de telemetría."]

        return ExplanationResponse(
            summary=f"Estado {data.level}: Puntuación de riesgo en {data.risk_score:.2f}.",
            recommended_actions=actions[:2],
            technical_details=f"Factores dominantes: {', '.join(data.top_factors)}. (Generado por motor de reglas)"
        )

    def explain(self, data: ExplanationRequest) -> ExplanationResponse:
        if not self.has_llm:
            return self._rule_based_fallback(data)

        prompt = self._build_prompt(data)
        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", "200",
            "--temp", "0.1",
            "-c", "1024"
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            out = res.stdout.strip()
            json_start = out.find("{")
            json_end = out.rfind("}") + 1
            parsed = json.loads(out[json_start:json_end])
            return ExplanationResponse(**parsed)
        except Exception:
            return self._rule_based_fallback(data)