# GridZ Shield — OsosMaduros_IBM Z Datathon 2026 Project

**IBM Z Datathon 2026 · Event code: SSF2026 · Theme: Real-time AI for critical decisions**

GridZ Shield is an explainable AI decision-support tool for power-grid operators. It estimates how much **inertia** the Spanish peninsular grid has, forecasts tomorrow's frequency-stability risk as a **Grid Risk Index** (`Stable` / `Watch` / `Critical`), and uses **IBM Granite** to explain the risk in plain language with two recommended actions. Everything — data processing, model training, inference and LLM text generation — runs inside a container on **IBM LinuxONE (s390x)**.

> GridZ Shield supports decisions; it never controls the grid.

![GridZ Shield architecture](docs/architecture.png)

## The problem
On **28 April 2025** the Iberian Peninsula suffered a complete blackout. As wind and solar replace conventional plants, the grid loses *inertia*: the kinetic energy of the large synchronous generators that slows down frequency changes when something fails. With less inertia, the frequency falls faster after a disturbance (higher RoCoF, *rate of change of frequency*) and operators have less time to react.

Operators need to know **when the grid is entering a low-inertia period**, and why, so they can prepare in advance (for example, by scheduling additional synchronous generation or reserves).

## What GridZ Shield does
1. **Ingests official data** from Red Eléctrica's public REData API: daily generation by technology, demand and cross-border exchanges for the peninsular system, 2021–2025 (1,826 days).
2. **Estimates inertia and RoCoF from physics.** Each technology gets a typical inertia constant *H* (nuclear 6 s, gas 4.3 s, hydro 4 s, solar thermal 3 s, wind and PV 0 s). System kinetic energy is `E = Σ Hᵢ·Pᵢ`, and the expected initial RoCoF after losing a 1,000 MW unit is `RoCoF = ΔP·f₀ / (2·E)`.
3. **Forecasts tomorrow's risk.** A scikit-learn RandomForest predicts the next day's estimated RoCoF; an IsolationForest flags unusual days. The prediction is mapped to the Grid Risk Index with thresholds learned from 2021–2024 (75th and 95th percentiles).
4. **Explains it with Granite.** IBM Granite 3.3 2B Instruct, running locally on LinuxONE through llama.cpp, writes a one-sentence summary and two actions for the operator. All figures are computed in code and passed to the model, so it cannot invent numbers.
5. **Shows it in a dashboard.** A replay of the two weeks before the blackout (14–28 April 2025), served by the same API.

## How GridZ Shield uses IBM LinuxONE
| Layer | What runs on LinuxONE |
|---|---|
| Platform | LinuxONE Community Cloud VM, **s390x (IBM z17)**, 3 vCPU, Docker container from the Jupyter Lab ML image |
| Data | `data/ingest_apis.py` downloads and processes REData inside the VM; the Parquet file is written with `fastparquet` and read back to verify it on a big-endian system |
| Predictive AI | `model/train.py` trains the scikit-learn models **inside the VM** (`grid_model.joblib`) |
| Inference | FastAPI + uvicorn service in the container, bound to `127.0.0.1` (not exposed to the internet); `/health` reports the real CPU architecture (`s390x`) |
| Generative AI | IBM Granite 3.3 2B Instruct in **big-endian GGUF** format, executed with `llama-cpp-python`; the container's llama.cpp build includes the **zDNN backend for the IBM Z NNPA accelerator** |
| Security | No data, prompts or model outputs leave the VM; no external LLM APIs; no secrets in the repository |

This combines the two AI patterns of the datathon labs: **Predictive AI (multi-model)** — regression plus anomaly detection — and **Predictive AI + Generative AI** — Granite explaining the prediction.

## Results
Evaluated on 2025, a year the model never saw during training (362 days):

| Metric | Value |
|---|---|
| Next-day RoCoF error (MAE), GridZ Shield | 0.0222 Hz/s |
| Next-day RoCoF error (MAE), baseline "tomorrow = today" | 0.0251 Hz/s |
| Improvement over the baseline | **11.6 %** |
| Risk level (Stable / Watch / Critical) matching the actual level | 82.9 % |
| Actual `Critical` days also forecast as `Critical` | 44 % (11 of 25) |

**Blackout case study (retrospective).** In the replay, the Easter week before the blackout (16–21 April 2025, low demand and high solar) reaches `Critical`, and most of the two weeks before 28 April stay in `Watch` or `Critical`. The forecast made on 26 April for the 27th is `Critical` (estimated inertia 57 GW·s against a typical 73–94 GW·s, with Spain exporting about 3.3 GW).

**Data checks.**
- Daily totals were cross-checked by hand against the REData API (for example, 5 January 2024).
- Generation minus net exports matches demand within about 2 % on average (the gap is mainly pumped-storage consumption).
- The implied system inertia constant (≈ 2.8 s on average) is of the same order as published estimates for the peninsular system (≈ 2.3 s; ENTSO-E range 2.14–2.64 s).

### Limitations (stated honestly)
- **Daily resolution.** Open data without a token is daily, so GridZ Shield flags low-inertia *periods* a day ahead; it does not predict events minutes ahead. The architecture accepts higher-resolution data (for example, ESIOS 5-minute data) without changes to the service.
- **Estimates, not measurements.** Inertia and RoCoF are physics-based estimates from the generation mix; measured grid frequency is not publicly available.
- **We do not claim to have predicted the blackout.** Post-event analyses point to a voltage-control problem as the trigger; low inertia is a vulnerability factor. There is a single documented event in the data, so it is used for a retrospective case study, not for training.
- **Granite on CPU is slow.** Each explanation takes several minutes on the 3-vCPU VM, so the dashboard uses explanations pre-generated on LinuxONE (`app/pregenerate.py`) and can also generate one live on demand.

## Run it on LinuxONE
All commands run in the **Jupyter Lab terminal** of the VM (inside the container), from the repository root.

```bash
# 0. Check the environment (libraries are already in the container image)
uname -m                                   # s390x
python3 -c "import pandas, sklearn, fastparquet, fastapi, uvicorn, llama_cpp; print('OK')"

# 1. Granite 3.3 2B Instruct, big-endian (1.55 GB, once)
mkdir -p ~/shared/models
wget -c -P ~/shared/models https://huggingface.co/taronaeo/Granite-3.3-2B-Instruct-BE-GGUF/resolve/main/granite-3.3-2b-instruct-be.Q4_K_M.gguf

# 2. Data and model (about 3 minutes)
python3 data/ingest_apis.py
python3 model/train.py

# 3. Pre-generate Granite explanations for the demo days (several minutes per day, resumable)
python3 -m app.pregenerate --start 2025-04-24 --end 2025-04-28

# 4. Start the API and the dashboard
nohup python3 -m uvicorn service.main:app --host 127.0.0.1 --port 8000 > api.log 2>&1 &
curl -s http://127.0.0.1:8000/health
```

Open the dashboard from your computer through an SSH tunnel (`ssh -i <key.pem> -L 8000:127.0.0.1:8000 <user>@<vm-ip>`) at `http://localhost:8000/app?autoplay=1`.

Configuration (environment variables): `GRID_MODEL_PATH` (default `model/grid_model.joblib`), `GRANITE_MODEL_PATH` (default `~/shared/models/granite-3.3-2b-instruct-be.Q4_K_M.gguf`), `LLM_THREADS` (default: all CPUs on machines with up to 4, otherwise half).

### Local development (x86)
The same code runs on a laptop with `pip install -r requirements.txt` and a standard (little-endian) Granite GGUF, for example [bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF](https://huggingface.co/bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF), set through `GRANITE_MODEL_PATH`. Big-endian models only load on s390x.

## Repository layout
| Folder | Content |
|---|---|
| `data/` | `ingest_apis.py`: REData ingestion and feature building, including the inertia and RoCoF estimates |
| `model/` | `train.py`: next-day RoCoF regressor, anomaly detector and risk thresholds; `grid_model.joblib` |
| `service/` | FastAPI inference API: `/predict`, `/explain`, `/health` |
| `explain/` | Granite explanation layer (llama-cpp-python, GBNF-constrained JSON, rule-based fallback) |
| `app/` | Operator dashboard (`/app`), replay data and the Granite pre-generation script |
| `docs/` | Architecture diagram (`architecture.svg` / `.png`) and screenshots |

Module contracts (columns, JSON formats and endpoints) are in [ARCHITECTURE.md](ARCHITECTURE.md); project rules are in [CONTEXT.md](CONTEXT.md).

## Data sources
- [REData API — Red Eléctrica](https://www.ree.es/es/apidatos): `generacion/estructura-generacion`, `demanda/evolucion`, `intercambios/todas-fronteras-programados` (peninsular system, daily).
- Inertia constants per technology: OECD-NEA (2020), [*Operation and control of power systems with reduced synchronous inertia*](https://oecd-nea.org/upload/docs/application/pdf/2020-07/workshop_esr_4._operation_and_control_of_power_systems_with_reduced_synchronous_inertia.pdf), slide 25 (Continental Europe, EUCO30 and TYNDP2018).
- Reference value for the peninsular system inertia used as a sanity check: L. Rouco, [ICREPQ keynote](https://icrepq.com/Santander/Keynotes/PL3-Rouco.pdf).

## Open-source libraries and resources used
| Resource | Use | Link |
|---|---|---|
| pandas | Data processing | https://pandas.pydata.org |
| NumPy | Numerical computing | https://numpy.org |
| Requests | REData API client | https://requests.readthedocs.io |
| fastparquet | Parquet I/O on s390x | https://github.com/dask/fastparquet |
| scikit-learn | RandomForest regressor, IsolationForest | https://scikit-learn.org |
| joblib | Model persistence | https://joblib.readthedocs.io |
| FastAPI | Inference API | https://fastapi.tiangolo.com |
| Uvicorn | ASGI server | https://github.com/encode/uvicorn |
| Pydantic | API schemas and validation | https://docs.pydantic.dev |
| python-dotenv | Environment configuration | https://github.com/theskumar/python-dotenv |
| llama-cpp-python | Granite inference from Python | https://github.com/abetlen/llama-cpp-python |
| llama.cpp | LLM runtime (s390x, zDNN backend) | https://github.com/ggml-org/llama.cpp |
| IBM Granite 3.3 2B Instruct | Explanation model (Apache 2.0) | https://huggingface.co/ibm-granite/granite-3.3-2b-instruct |
| Granite 3.3 2B Instruct big-endian GGUF | s390x build of the model | https://huggingface.co/taronaeo/Granite-3.3-2B-Instruct-BE-GGUF |
| Chart.js | Dashboard charts (loaded by the browser) | https://www.chartjs.org |
| LinuxONE Community Cloud Jupyter Lab ML | Container image used on the VM | https://github.com/linuxone-community-cloud/jupyter-lab-ml |

## Team — OsosMaduros
David Armas · Adrián Kearns · Jaime Monjas · Héctor Pérez

## License
MIT — see [LICENSE](LICENSE).
