"""Mini-script de prueba asíncrono del pipeline."""

import asyncio

from chain import process_text

TEXTOS = {
    "Log de error (caso claro)": (
        "ERROR 2026-11-12 14:03:22 - La API en FastAPI devuelve 503 desde hace 20 "
        "minutos. Redis alcanzó el límite de memoria y las consultas a PostgreSQL "
        "superan el pool de 100 conexiones. Los usuarios no pueden hacer checkout."
    ),
    "Arquitectura (caso informativo)": (
        "El frontend está hecho en React y consume un backend en Django REST. "
        "Desplegamos con Docker en AWS ECS y usamos GitHub Actions para CI."
    ),
    "Prueba de estrés (texto ambiguo, sin tecnología)": (
        "Hoy almorcé milanesas con puré y después fui a caminar al parque."
    ),
}


async def main() -> None:
    for titulo, texto in TEXTOS.items():
        print(f"\n===== {titulo} =====")
        try:
            resultado = await process_text(texto)
            print(resultado.model_dump_json(indent=2))
        except Exception as e:
            print(f"No se pudo extraer: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
