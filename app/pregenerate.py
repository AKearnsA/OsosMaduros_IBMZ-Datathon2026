"""
Pregenera con Granite (en LinuxONE) las explicaciones de los dias del replay y las
guarda en app/explanations.json. En la VM cada explicacion tarda varios minutos,
asi que se generan una vez y el dashboard las muestra al instante.

Uso (desde la raiz del repo, en la terminal de Jupyter):
    python3 -m app.pregenerate                                   # 14 al 28 de abril de 2025
    python3 -m app.pregenerate --start 2025-04-24 --end 2025-04-28

Es reanudable: si se corta, al relanzarlo salta los dias ya generados.
"""

import argparse
import json
import os
import platform
import time
from datetime import date, datetime, timezone

from app.replay import replay_days
from app.router import DEFAULT_END, DEFAULT_START, EXPLANATIONS_PATH
from explain.generator import GraniteExplainer
from explain.schemas import ExplanationRequest
from service.utils import GridPredictor

EXPLAIN_VALUES = [
    "inertia_est_gws",
    "nonsync_share",
    "interconnection_balance_mw",
    "rocof_est_hz_s",
    "demand_mw",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument(
        "--allow-rules",
        action="store_true",
        help="guardar tambien explicaciones por reglas si Granite no esta disponible",
    )
    args = parser.parse_args()

    predictor = GridPredictor()
    explainer = GraniteExplainer()
    if not explainer.available and not args.allow_rules:
        raise SystemExit(f"[!] Granite no disponible ({explainer.load_error}). Abortado.")

    stored = {}
    if EXPLANATIONS_PATH.exists():
        stored = json.loads(EXPLANATIONS_PATH.read_text(encoding="utf-8"))

    days = replay_days(args.start, args.end)
    print(f"[*] {len(days)} dias en {platform.machine()} | ya generados: {len(stored)}")

    for i, day in enumerate(days, 1):
        key = day["forecast_date"]
        if key in stored and stored[key].get("source") == "granite":
            print(f"[{i}/{len(days)}] {key}: ya generada, se salta")
            continue

        risk_score, level, top_factors = predictor.predict(
            {"timestamp": day["timestamp"], **day["features"]}
        )
        request = ExplanationRequest(
            timestamp=day["timestamp"],
            risk_score=risk_score,
            level=level,
            top_factors=top_factors,
            **{k: day["features"][k] for k in EXPLAIN_VALUES},
        )

        t0 = time.time()
        explanation = explainer.explain(request)
        seconds = round(time.time() - t0, 1)

        stored[key] = {
            **explanation.model_dump(),
            "level": level,
            "risk_score": risk_score,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "arch": platform.machine(),
            "model": os.path.basename(explainer.model_path),
            "seconds": seconds,
        }
        # Se guarda tras cada dia para poder reanudar si se corta
        EXPLANATIONS_PATH.write_text(
            json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{i}/{len(days)}] {key}: {level} -> {explanation.source} en {seconds}s")

    print(f"[OK] Explicaciones guardadas en {EXPLANATIONS_PATH}")


if __name__ == "__main__":
    main()
