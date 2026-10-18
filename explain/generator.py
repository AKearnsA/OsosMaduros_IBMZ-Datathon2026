import os
import json
import threading
from explain.schemas import ExplanationRequest, ExplanationResponse

# Granite 3.3 2B Instruct en big-endian para s390x (llama.cpp / llama-cpp-python).
# Modelo: https://huggingface.co/taronaeo/Granite-3.3-2B-Instruct-BE-GGUF
DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/shared/models/granite-3.3-2b-instruct-be.Q4_K_M.gguf"
)

# Granite solo redacta el resumen y las 2 acciones (lo que aporta lenguaje natural).
# technical_details se construye en codigo con las cifras exactas: mas rapido y sin
# riesgo de cifras inventadas.
GBNF_GRAMMAR = r'''
root ::= "{" ws "\"summary\":" ws string "," ws "\"recommended_actions\":" ws "[" ws string "," ws string ws "]" ws "}"
string ::= "\"" ([^"\\\n] | "\\" ["\\/bfnrt])* "\""
ws ::= [ \t\n]*
'''

# Hilos de CPU para Granite (configurable; la mitad de los nucleos suele ir mas rapido).
LLM_THREADS = int(os.getenv("LLM_THREADS", max(1, (os.cpu_count() or 2) // 2)))
CACHE_MAX_ITEMS = 256

LEVEL_GUIDANCE = {
    "Stable": "La red esta estable: dilo claramente y no hables de riesgo alto. "
              "Las acciones son de supervision rutinaria.",
    "Watch": "Riesgo MODERADO (no alto) de inestabilidad: recomienda vigilancia y preparacion.",
    "Critical": "Riesgo alto de inestabilidad de frecuencia: recomienda acciones preventivas.",
}

FACTOR_LABELS = {
    "inertia_est_gws": "inercia estimada baja",
    "nonsync_share": "cuota alta de eolica y solar",
    "interconnection_balance_mw": "saldo de interconexion elevado",
}

SYSTEM_PROMPT = (
    "Eres el asistente de GridZ Shield, que apoya a los operadores de la red electrica "
    "peninsular espanola. Solo das apoyo a la decision: nunca controlas la red. "
    "Explicas el riesgo de baja inercia de forma breve y clara, en espanol. "
    "Usa unicamente las cifras que te dan; no inventes datos. "
    "Responde en JSON con: summary (una sola frase corta que explique por que hay ese "
    "nivel de riesgo) y recommended_actions (exactamente 2 acciones concretas para el "
    "operador, de menos de 15 palabras cada una)."
)


def _band(value: float, low: float, high: float, low_label: str, high_label: str) -> str:
    """Clasifica un valor frente a los percentiles 25 y 75 de 2021-2024."""
    if value < low:
        return low_label
    if value > high:
        return high_label
    return "NORMAL"


class GraniteExplainer:
    def __init__(
        self,
        model_path: str = os.getenv("GRANITE_MODEL_PATH", DEFAULT_MODEL_PATH),
        n_ctx: int = 1024,
    ):
        self.model_path = model_path
        self.llm = None
        self.grammar = None
        self.load_error = None
        # llama.cpp no admite llamadas concurrentes sobre el mismo modelo
        self._lock = threading.Lock()
        # Misma peticion -> misma explicacion (el replay de la demo no espera dos veces)
        self._cache: dict = {}

        if not os.path.exists(model_path):
            self.load_error = f"No se encontro el modelo Granite en {model_path}"
            print(f"[!] {self.load_error}. /explain usara reglas.")
            return

        try:
            from llama_cpp import Llama, LlamaGrammar

            # El modelo se carga una sola vez, al arrancar la API.
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=LLM_THREADS,
                verbose=False,
            )
            self.grammar = LlamaGrammar.from_string(GBNF_GRAMMAR, verbose=False)
            print(f"[OK] Granite cargado: {model_path}")
        except Exception as e:
            self.llm = None
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"[!] Granite NO cargado -> {self.load_error}. /explain usara reglas.")

    @property
    def available(self) -> bool:
        return self.llm is not None

    def _build_user_prompt(self, data: ExplanationRequest) -> str:
        lines = [
            f"Nivel de riesgo previsto para manana: {data.level} "
            f"(puntuacion {data.risk_score:.2f} sobre 1).",
        ]
        if data.top_factors:
            labels = [FACTOR_LABELS.get(f, f) for f in data.top_factors]
            lines.append(f"Factores principales: {', '.join(labels)}.")
        else:
            lines.append("Ningun factor esta fuera de lo normal.")

        # Las comparaciones se hacen aqui (percentiles 2021-2024): un modelo de 2B
        # interpreta mal los numeros, asi que recibe la conclusion ya hecha.
        if data.inertia_est_gws is not None:
            lines.append(
                f"Inercia estimada: {data.inertia_est_gws:.1f} GW*s, "
                f"{_band(data.inertia_est_gws, 73.5, 94.1, 'BAJA (peor de lo normal)', 'ALTA (mejor de lo normal)')}."
            )
        if data.nonsync_share is not None:
            lines.append(
                f"Generacion eolica y solar: {data.nonsync_share:.0%} del total, "
                f"{_band(data.nonsync_share, 0.304, 0.464, 'BAJA', 'ALTA')}."
            )
        if data.rocof_est_hz_s is not None:
            lines.append(
                f"RoCoF estimado si se perdiera de golpe una central de 1.000 MW "
                f"(escenario hipotetico, no ha ocurrido): {data.rocof_est_hz_s:.3f} Hz/s, "
                f"{_band(data.rocof_est_hz_s, 0.266, 0.340, 'BAJO (mejor de lo normal)', 'ALTO (peor de lo normal)')}."
            )
        if data.interconnection_balance_mw is not None:
            sentido = "EXPORTA" if data.interconnection_balance_mw < 0 else "IMPORTA"
            volumen = "elevado" if abs(data.interconnection_balance_mw) > 2630 else "normal"
            lines.append(
                f"Intercambio internacional: Espana {sentido} {abs(data.interconnection_balance_mw):.0f} MW (volumen {volumen})."
            )
        if data.demand_mw is not None:
            lines.append(f"Demanda media: {data.demand_mw:.0f} MW.")
        if data.is_simulated:
            lines.append("Atencion: son datos simulados.")
        lines.append(LEVEL_GUIDANCE.get(data.level, ""))
        return "\n".join(lines)

    @staticmethod
    def _technical_details(data: ExplanationRequest) -> str:
        """Cifras exactas del dia, sin pasar por el LLM."""
        parts = []
        if data.inertia_est_gws is not None:
            parts.append(f"Inercia estimada {data.inertia_est_gws:.1f} GW·s")
        if data.rocof_est_hz_s is not None:
            parts.append(f"RoCoF estimado {data.rocof_est_hz_s:.3f} Hz/s ante la pérdida de 1.000 MW")
        if data.nonsync_share is not None:
            parts.append(f"eólica y solar {data.nonsync_share:.0%} de la generación")
        if data.interconnection_balance_mw is not None:
            sentido = "exportación" if data.interconnection_balance_mw < 0 else "importación"
            parts.append(f"saldo de interconexión {abs(data.interconnection_balance_mw):.0f} MW en {sentido}")
        factors = ", ".join(data.top_factors) or "ninguno fuera de lo normal"
        detail = f"Factores dominantes: {factors}."
        if parts:
            detail += " " + "; ".join(parts) + "."
        return detail

    def _rule_based_fallback(self, data: ExplanationRequest, reason: str) -> ExplanationResponse:
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
            technical_details=f"{self._technical_details(data)} (Modo reglas: {reason})",
            source="rules",
        )

    def explain(self, data: ExplanationRequest) -> ExplanationResponse:
        if not self.available:
            return self._rule_based_fallback(data, self.load_error or "Granite no disponible")

        cache_key = data.model_dump_json()
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with self._lock:
                result = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": self._build_user_prompt(data)},
                    ],
                    grammar=self.grammar,
                    max_tokens=200,
                    temperature=0.2,
                )
            parsed = json.loads(result["choices"][0]["message"]["content"])
            response = ExplanationResponse(
                summary=parsed["summary"],
                recommended_actions=parsed["recommended_actions"],
                technical_details=self._technical_details(data),
                source="granite",
            )
            if len(self._cache) >= CACHE_MAX_ITEMS:
                self._cache.clear()
            self._cache[cache_key] = response
            return response
        except Exception as e:
            return self._rule_based_fallback(data, f"fallo de Granite ({type(e).__name__}: {e})")
