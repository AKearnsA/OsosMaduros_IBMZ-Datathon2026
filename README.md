# GridZ Shield

**IBM Z Datathon 2026 - Event code: SSF2026**
Theme: Real-time AI for critical decisions

GridZ Shield is a real-time AI system that detects early signs of power-grid instability as renewable generation grows, and predicts the risk of frequency events minutes ahead. It gives grid operators an explainable Grid Risk Index (Stable / Watch / Critical) for decision support. It runs on IBM LinuxONE.

> Status: repository scaffolding only. All project code is written during the datathon window (17-18 Oct 2026).

## Team
- Team name: Osos Maduros
- Members: David Armas,Adrián Kearns,Jaime Monjas,Héctor Pérez

## Repository layout
| Folder | Purpose |
|---|---|
| `data/` | Data ingestion and feature building (ESIOS, REData, ENTSO-E) |
| `model/` | Anomaly detection and risk prediction models |
| `service/` | Inference service running on the LinuxONE VM |
| `explain/` | LLM layer that explains alerts to operators (runs in the VM) |
| `app/` | Demo dashboard with historical replay |
| `docs/` | Architecture diagram, screenshots of the LinuxONE instance |

See [CONTEXT.md](CONTEXT.md) for project rules and [ARCHITECTURE.md](ARCHITECTURE.md) for the module contracts.

## Data sources (public, openly licensed)
- ESIOS / Red Electrica de Espana API
- REData (REE)
- ENTSO-E Transparency Platform
- Public reports on the April 2025 Iberian blackout (retrospective case study)

## Open-source libraries and resources used
_To be completed during the event (links required by the rules)._

## IBM Z / LinuxONE usage
_To be completed: tech stack, screenshots of the instance, model deployment._

## License
MIT - see [LICENSE](LICENSE).
