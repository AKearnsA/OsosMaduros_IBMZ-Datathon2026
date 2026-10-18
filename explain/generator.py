import os
import json
import threading
from explain.schemas import ExplanationRequest, ExplanationResponse

# Granite 3.3 2B Instruct big-endian for s390x (llama.cpp / llama-cpp-python).
# Model: https://huggingface.co/taronaeo/Granite-3.3-2B-Instruct-BE-GGUF
DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/shared/models/granite-3.3-2b-instruct-be.Q4_K_M.gguf"
)

# Granite only writes the summary and the 2 actions (the natural-language part).
# technical_details is built in code with the exact figures: faster and with no
# risk of invented numbers.
GBNF_GRAMMAR = r'''
root ::= "{" ws "\"summary\":" ws string "," ws "\"recommended_actions\":" ws "[" ws string "," ws string ws "]" ws "}"
string ::= "\"" ([^"\\\n] | "\\" ["\\/bfnrt])* "\""
ws ::= [ \t\n]*
'''

# CPU threads for Granite (configurable with LLM_THREADS).
# On small machines (the LinuxONE VM has 3 CPUs, 1 thread per core) all are used;
# on laptops with hyperthreading, half is usually faster.
_CPUS = os.cpu_count() or 1
LLM_THREADS = int(os.getenv("LLM_THREADS", _CPUS if _CPUS <= 4 else _CPUS // 2))
CACHE_MAX_ITEMS = 256

LEVEL_GUIDANCE = {
    "Stable": "The grid is stable: say so clearly and do not talk about high risk. "
              "Actions are routine monitoring.",
    "Watch": "MODERATE risk (not high) of instability: recommend vigilance and preparation.",
    "Critical": "High risk of frequency instability: recommend preventive actions.",
}

FACTOR_LABELS = {
    "inertia_est_gws": "low estimated inertia",
    "nonsync_share": "high share of wind and solar",
    "interconnection_balance_mw": "high cross-border exchange",
}

SYSTEM_PROMPT = (
    "You are the GridZ Shield assistant, supporting operators of the Spanish peninsular "
    "power grid. You only provide decision support: you never control the grid. "
    "You explain the risk of low inertia briefly and clearly, in English. "
    "Use only the figures you are given; do not invent data. "
    "Answer in JSON with: summary (one short sentence explaining why the risk level is "
    "what it is) and recommended_actions (exactly 2 concrete actions for the "
    "operator, under 15 words each)."
)


def _band(value: float, low: float, high: float, low_label: str, high_label: str) -> str:
    """Classifies a value against the 25th and 75th percentiles of 2021-2024."""
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
        # llama.cpp does not support concurrent calls on the same model
        self._lock = threading.Lock()
        # Same request -> same explanation (the demo replay never waits twice)
        self._cache: dict = {}

        if not os.path.exists(model_path):
            self.load_error = f"Granite model not found at {model_path}"
            print(f"[!] {self.load_error}. /explain will use rules.")
            return

        try:
            from llama_cpp import Llama, LlamaGrammar

            # The model is loaded once, when the API starts.
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=LLM_THREADS,
                verbose=False,
            )
            self.grammar = LlamaGrammar.from_string(GBNF_GRAMMAR, verbose=False)
            print(f"[OK] Granite loaded: {model_path}")
        except Exception as e:
            self.llm = None
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"[!] Granite NOT loaded -> {self.load_error}. /explain will use rules.")

    @property
    def available(self) -> bool:
        return self.llm is not None

    def _build_user_prompt(self, data: ExplanationRequest) -> str:
        lines = [
            f"Predicted risk level for tomorrow: {data.level} "
            f"(score {data.risk_score:.2f} out of 1).",
        ]
        if data.top_factors:
            labels = [FACTOR_LABELS.get(f, f) for f in data.top_factors]
            lines.append(f"Main factors: {', '.join(labels)}.")
        else:
            lines.append("No factor is outside the norm.")

        # Comparisons are made here (2021-2024 percentiles): a 2B model
        # misreads numbers, so it receives the conclusion already drawn.
        if data.inertia_est_gws is not None:
            lines.append(
                f"Estimated inertia: {data.inertia_est_gws:.1f} GW*s, "
                f"{_band(data.inertia_est_gws, 73.5, 94.1, 'LOW (worse than normal)', 'HIGH (better than normal)')}."
            )
        if data.nonsync_share is not None:
            lines.append(
                f"Wind and solar generation: {data.nonsync_share:.0%} of the total, "
                f"{_band(data.nonsync_share, 0.304, 0.464, 'LOW', 'HIGH')}."
            )
        if data.rocof_est_hz_s is not None:
            lines.append(
                f"Estimated RoCoF if a 1,000 MW plant were suddenly lost "
                f"(hypothetical scenario, it has not happened): {data.rocof_est_hz_s:.3f} Hz/s, "
                f"{_band(data.rocof_est_hz_s, 0.266, 0.340, 'LOW (better than normal)', 'HIGH (worse than normal)')}."
            )
        if data.interconnection_balance_mw is not None:
            direction = "EXPORTS" if data.interconnection_balance_mw < 0 else "IMPORTS"
            volume = "high" if abs(data.interconnection_balance_mw) > 2630 else "normal"
            lines.append(
                f"Cross-border exchange: Spain {direction} {abs(data.interconnection_balance_mw):.0f} MW (volume {volume})."
            )
        if data.demand_mw is not None:
            lines.append(f"Average demand: {data.demand_mw:.0f} MW.")
        if data.is_simulated:
            lines.append("Warning: this is simulated data.")
        lines.append(LEVEL_GUIDANCE.get(data.level, ""))
        return "\n".join(lines)

    @staticmethod
    def _technical_details(data: ExplanationRequest) -> str:
        """Exact figures for the day, without going through the LLM."""
        parts = []
        if data.inertia_est_gws is not None:
            parts.append(f"Estimated inertia {data.inertia_est_gws:.1f} GW·s")
        if data.rocof_est_hz_s is not None:
            parts.append(f"estimated RoCoF {data.rocof_est_hz_s:.3f} Hz/s on a 1,000 MW loss")
        if data.nonsync_share is not None:
            parts.append(f"wind and solar {data.nonsync_share:.0%} of generation")
        if data.interconnection_balance_mw is not None:
            direction = "export" if data.interconnection_balance_mw < 0 else "import"
            parts.append(f"cross-border balance {abs(data.interconnection_balance_mw):.0f} MW ({direction})")
        factors = ", ".join(FACTOR_LABELS.get(f, f) for f in data.top_factors) or "none outside the norm"
        detail = f"Dominant factors: {factors}."
        if parts:
            detail += " " + "; ".join(parts) + "."
        return detail

    def _rule_based_fallback(self, data: ExplanationRequest, reason: str) -> ExplanationResponse:
        actions_map = {
            "nonsync_share": "Monitor inertia and prepare synchronous reserves.",
            "interconnection_balance_mw": "Adjust flows on the cross-border interconnection.",
            "inertia_est_gws": "Activate synthetic inertia or synchronous power."
        }
        actions = [actions_map[f] for f in data.top_factors if f in actions_map]
        if not actions:
            actions = ["Maintain routine telemetry monitoring."]

        return ExplanationResponse(
            summary=f"Status {data.level}: risk score {data.risk_score:.2f} out of 1.",
            recommended_actions=actions[:2],
            technical_details=f"{self._technical_details(data)} (Rules mode: Granite unavailable on this architecture)",
            source="rules",
        )

    def explain(self, data: ExplanationRequest) -> ExplanationResponse:
        if not self.available:
            return self._rule_based_fallback(data, self.load_error or "Granite unavailable")

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
            return self._rule_based_fallback(data, f"Granite failure ({type(e).__name__}: {e})")
            