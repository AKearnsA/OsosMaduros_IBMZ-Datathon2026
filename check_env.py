"""
GridZ Shield - environment check.

Shows the CPU architecture (s390x on IBM LinuxONE) and verifies that every library
the project needs can be imported. Run it from the repository root:

    python3 check_env.py
"""

import platform
import sys

REQUIRED = {
    "pandas": "data processing",
    "numpy": "numerical computing",
    "requests": "REData API client",
    "fastparquet": "features.parquet I/O",
    "sklearn": "RandomForest and IsolationForest models",
    "joblib": "model persistence",
    "fastapi": "inference API",
    "uvicorn": "ASGI server",
    "pydantic": "API schemas",
}
OPTIONAL = {
    "llama_cpp": "IBM Granite explanations (without it /explain uses rules)",
}


def check(packages: dict, required: bool) -> list:
    missing = []
    for name, purpose in packages.items():
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "installed")
            print(f"  OK       {name:<12} {version:<12} {purpose}")
        except ImportError:
            label = "MISSING" if required else "optional"
            print(f"  {label:<8} {name:<12} {'-':<12} {purpose}")
            missing.append(name)
    return missing


def main() -> None:
    arch = platform.machine()
    print("=" * 64)
    print("GRIDZ SHIELD - ENVIRONMENT CHECK")
    print("=" * 64)
    print(f"CPU architecture : {arch}{'  (IBM Z / LinuxONE)' if arch == 's390x' else ''}")
    print(f"Byte order       : {sys.byteorder}-endian")
    print(f"Python           : {sys.version.split()[0]}")

    print("\nRequired libraries")
    missing = check(REQUIRED, required=True)
    print("\nOptional libraries")
    check(OPTIONAL, required=False)

    print("=" * 64)
    if missing:
        print(f"Missing required libraries: {', '.join(missing)} (pip install -r requirements.txt)")
        sys.exit(1)
    print("Environment ready for GridZ Shield.")


if __name__ == "__main__":
    main()
