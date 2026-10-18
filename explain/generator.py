import os
import json
import re
import subprocess
from explain.schemas import ExplanationRequest, ExplanationResponse


# Gramática GBNF para obligar a llama.cpp a generar ÚNICAMENTE el JSON esperado
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
        binary_path: str = os.getenv("LLAMA_BINARY_PATH", "./llama.cpp/llama-cli")
    ):
        self.model_path = model_path
        # Detectar binario automáticamente entre llama-cli y main
        if not os.path.exists(binary_path) and os.path.exists("./llama.cpp/main"):
            self.binary_path = "./llama.cpp/main"
        else:
            self.binary_path = binary_path

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

    def explain(self, data: ExplanationRequest) -> ExplanationResponse:
        prompt = self._build_prompt(data)

        # Comando optimizado para ejecución directa del LLM en s390x
        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", "250",
            "--temp", "0.1",
            "-c", "2048",
            "--grammar", GBNF_GRAMMAR,  # Obliga al LLM a seguir el esquema JSON estrictamente
            "--log-disable"             # Apaga logs de inicialización en stdout
        ]

        # Ejecutamos el subproceso capturando la salida limpia
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = res.stdout.strip()

        # Extraer el bloque JSON de la salida mediante expresión regular
        match = re.search(r'\{.*\}', out, re.DOTALL)
        if not match:
            raise RuntimeError(f"Error de generación en LLM. Salida de stdout: {out}\nStderr: {res.stderr}")

        json_str = match.group(0)
        parsed = json.loads(json_str)

        return ExplanationResponse(**parsed)
        