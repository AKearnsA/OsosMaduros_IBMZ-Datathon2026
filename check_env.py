import sys
import platform

print("=" * 50)
print("GRIDZ SHIELD - COMPROBACIÓN DE ENTORNO")
print("=" * 50)

print(f"[+] Arquitectura de la CPU: {platform.machine()}")
print(f"[+] Versión de Python: {sys.version.split()[0]}")

packages = {
    "pandas": "Core Data Processing",
    "fastparquet": "Soporte Parquet (features.parquet)",
    "numpy": "Cálculos numéricos",
    "fastapi": "API de inferencia",
    "uvicorn": "Servidor ASGI",
    "sklearn": "Modelos ML (scikit-learn)",
    "joblib": "Persistencia del modelo",
    "pydantic": "Validación de esquemas JSON",
    "dotenv": "Gestión de variables de entorno (python-dotenv)",
    "requests": "Ingesta de APIs (ESIOS, REData)",
}

print("\n--- Comprobación de Dependencias ---")
missing = []

for pkg, desc in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "Instalada")
        print(f"  ✓ {pkg:<15} ({ver:<10}) -> {desc}")
    except ImportError:
        print(f"  ❌ {pkg:<15} (NO ENCONTRADA) -> {desc}")
        missing.append(pkg)

print("=" * 50)
if not missing:
    print("¡Entorno 100% verificado y listo para GridZ Shield!")
