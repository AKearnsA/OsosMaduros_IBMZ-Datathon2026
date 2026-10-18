import os
import json
import re
import subprocess
from explain.schemas import ExplanationRequest, ExplanationResponse

GBNF_GRAMMAR = r'''
root ::= "{" ws "\"summary\":" ws string "," ws "\"recommended_actions\":" ws string-array "," ws "\"technical_details\":" ws string "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
string-array ::= "[" ws (string ("," ws string)*)? ws "]"
ws ::= [ \t\n]*
'''

class GraniteExplainer:
    def __init__(
        self,
        model_path: str = os.getenv("GRANITE_MODEL_PATH", "models/granite-3.0-3b-instruct.Q4_K_M.gguf"),
        binary_path: str = os.getenv("LLAMA_BINARY_PATH", "")
    ):
        self.model_path = model_path
        
        # Búsqueda automática del binario de llama.cpp
        possible_paths = [
            binary_path,
            "./llama.cpp/llama-cli",
            "./llama.cpp/main",
            "./main",
            "/usr/local/bin/llama-cli"
        ]
        
        self.binary_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                self.binary_path = path
                break

    def _build_prompt(self, data: ExplanationRequest) -> str:
        factors_str = ", ".join(data.top_factors)
        return (
            f"<|system|>\n"
            f"Eres el copiloto de IA de GridZ Shield corriendo en IBM LinuxONE.\n"
            f"Tu tarea es analizar el riesgo de inestabilidad en la red eléctrica y generar un diagnóstico técnico breve con exactamente 2 acciones recomendadas para los operadores.\n"
            f"<|user|>\n"
            f"Lectura de Telemetría:\n"
            f"- Nivel de Riesgo: {data.level} (Puntuación: {data.risk_score:.2f})\n"
            f"- Factores Principales: {factors_str}\n"
            f"- Simulación: {data.is_simulated}\n"
            f"<|assistant|>\n"
        )

    def _rule_based_fallback(self, data: ExplanationRequest) -> ExplanationResponse:
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
            technical_details=f"Factores dominantes: {', '.join(data.top_factors)}. (Modo Contingencia - Sin Binario LLM)"
        )

    def explain(self, data: ExplanationRequest) -> ExplanationResponse:
        # Si no existe el binario o el modelo .gguf, usaremos la lógica determinista segura
        if not self.binary_path or not os.path.exists(self.model_path):
            return self._rule_based_fallback(data)

        prompt = self._build_prompt(data)

        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", "250",
            "--temp", "0.1",
            "-c", "2048",
            "--grammar", GBNF_GRAMMAR,
            "--log-disable"
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = res.stdout.strip()

            match = re.search(r'\{.*\}', out, re.DOTALL)
            if not match:
                return self._rule_based_fallback(data)

            json_str = match.group(0)
            parsed = json.loads(json_str)
            return ExplanationResponse(**parsed)
        except Exception:
            return self._rule_based_fallback(data)
            