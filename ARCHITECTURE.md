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

## features.parquet (proposed columns)
| Column | Type | Unit / meaning |
|---|---|---|
| `timestamp` | datetime (UTC) | Start of the interval |
| `demand_mw` | float | Demand |
| `gen_total_mw` | float | Total generation |
| `gen_nonsync_mw` | float | Wind + solar PV (non-synchronous) |
| `gen_sync_mw` | float | Nuclear, hydro, gas, etc. (synchronous) |
| `nonsync_share` | float | `gen_nonsync_mw / gen_total_mw`, range 0-1 |
| `interconnection_balance_mw` | float | Net imports (+) / exports (-) |
| `frequency_hz` | float, nullable | Only if available at sufficient resolution |
| `rocof_hz_s` | float, nullable | Only computed from suitable frequency data |
| `label_event` | int, nullable | 1 if an observed instability event; keep proxy labels in a separate column `label_proxy` |

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
