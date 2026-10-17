# Architecture and module contracts

Planning document. The column names and formats below are a proposal: agree on them as a team and edit here before writing code. Once fixed, they are the contract between modules.

## Data flow
data -> features file -> model -> inference service -> explanation -> dashboard

## Modules
| Module | Folder | Input | Output |
|---|---|---|---|
| Data | `data/` | ESIOS / REData / ENTSO-E APIs | `features.parquet` (columns below) |
| Model | `model/` | `features.parquet` | Saved model (`.joblib` or ONNX) + metrics file |
| Inference service | `service/` | One row of features | JSON risk result (below) |
| Explanation | `explain/` | The JSON risk result | Short plain-language text for the operator |
| Dashboard / demo | `app/` | Historical replay + service | Live-style view for the demo video |

## features.parquet (columns)
Source: REData (Red Electrica), peninsular system, daily resolution (MWh/day converted to average MW). Written with `fastparquet`.

| Column | Type | Unit / meaning |
|---|---|---|
| `timestamp` | string, ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`) | Start of the interval (local midnight for daily data) |
| `demand_mw` | float | Demand (REData `demanda/evolucion`) |
| `gen_total_mw` | float | Total generation (sum of technologies, excluding the API's own "Generacion total" row) |
| `gen_nonsync_mw` | float | Wind + solar PV + solar thermal (non-synchronous, team decision) |
| `gen_sync_mw` | float | Nuclear, hydro, gas, coal, etc. (synchronous) |
| `nonsync_share` | float | `gen_nonsync_mw / gen_total_mw`, range 0-1 |
| `interconnection_balance_mw` | float | Scheduled net imports (+) / exports (-), all borders |
| `label_event` | int, nullable | 1 only for an observed instability event (28 Apr 2025 blackout); null otherwise. Do not train on it |
| `label_proxy` | int (0/1) | Proxy label: 1 if `nonsync_share > 0.55`. Training target |
| `inertia_est_gws` | float | Estimated kinetic energy, sum(H_i * P_i) in GW*s (H per technology, see `data/ingest_apis.py`) |
| `rocof_est_hz_s` | float | Estimated initial RoCoF for a 1,000 MW loss with the peninsular system isolated: dP * f0 / (2 * E) |

Measured frequency and RoCoF (`frequency_hz`, `rocof_hz_s`) were dropped: no open source publishes them without a token. `inertia_est_gws` and `rocof_est_hz_s` are physical estimates, not measurements.

## Inference result (JSON)
```json
{
  "timestamp": "2025-04-28T10:30:00Z",
  "risk_score": 0.0,
  "level": "Stable",
  "top_factors": ["nonsync_share", "interconnection_balance_mw"],
  "is_simulated": false
}
```
`level` is one of `Stable`, `Watch`, `Critical`. `risk_score` is in [0, 1].

## Integration plan
First 3-4 hours: end-to-end skeleton with fake data (sample parquet -> fixed score -> dashboard). Then each owner replaces their part with the real implementation.

## Ownership (fill in)
| Module | Owner | Branch |
|---|---|---|
| data | _TODO_ | |
| model | _TODO_ | |
| service | _TODO_ | |
| explain | _TODO_ | |
| app | _TODO_ | |
