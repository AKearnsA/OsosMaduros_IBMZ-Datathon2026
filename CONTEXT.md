# Project context (read this first - for humans and AI assistants)

## Goal
GridZ Shield: real-time AI on IBM LinuxONE (s390x) that detects early signs of power-grid instability and predicts frequency-event risk minutes ahead, giving operators an explainable Grid Risk Index (Stable / Watch / Critical). Decision support only - no autonomous grid control.

## Hard rules (datathon)
- All code is written between 17 and 18 Oct 2026. Do not add project code before the kick-off.
- Code must be original; open-source libraries are allowed but must be linked in README.
- Never commit secrets: SSH keys (.pem), API tokens (ESIOS, ENTSO-E), Jupyter token, VM IP.
- The explanation LLM must run inside the LinuxONE VM (e.g. Granite via llama.cpp), not via external APIs.

## Environment
- Runtime: Linux on IBM LinuxONE (s390x), Python 3, Jupyter Lab container.
- Before adding a dependency, check it installs on s390x. Announce new dependencies to the team and add them to the shared `requirements.txt`.

## Conventions
- Python 3 with type hints; format with `black`.
- Each module lives in its own folder and is edited only by its owner. Work on a personal branch and open a pull request to `main`.
- Each module includes a minimal example (sample input + a command that runs it).
- Use exactly the column names and JSON format defined in ARCHITECTURE.md. Do not rename them.

## Data principles
- Distinguish observed events from proxy labels.
- Simulated disturbances are labelled and reported separately from real-event evaluation.
- RoCoF is computed only from frequency records with sufficient resolution; otherwise use the synchronous/non-synchronous generation share as an inertia proxy.
