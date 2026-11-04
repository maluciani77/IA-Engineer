"""Script de validacion: modo normal y modo streaming.

    python main.py              # usa LLM_PROVIDER del .env
    python main.py openai
    python main.py anthropic
    python main.py both

Devuelve 0 si todo lo que se probo funciono, 1 si algo fallo o si no habia
ninguna API key cargada.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from manager import AsyncLLMManager
from schemas import ChatMessage, Provider

# La consola de Windows suele venir en cp1252 y destroza los acentos.
for flujo in (sys.stdout, sys.stderr):
    flujo.reconfigure(encoding="utf-8", errors="replace")

PREGUNTA = "¿Que es la entropia? Responde en dos oraciones."

SYSTEM = ChatMessage.system("Sos un profesor de fisica. Se breve y concreto.")

API_KEY_ENV: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


async def modo_normal(manager: AsyncLLMManager) -> bool:
    """Una respuesta completa. Devuelve si salio bien."""
    print("\n--- modo normal ---")
    respuesta = await manager.generate([SYSTEM, ChatMessage.user(PREGUNTA)])
    if not respuesta.ok:
        print(f"error controlado -> {respuesta.error}")
        return False
    print(respuesta.content.strip())
    tokens = respuesta.usage.total_tokens if respuesta.usage else "?"
    print(f"\n[{respuesta.model} | {respuesta.latency_ms} ms | {tokens} tokens]")
    return True


async def modo_streaming(manager: AsyncLLMManager) -> bool:
    """Imprime los tokens a medida que llegan. Devuelve si salio bien."""
    print("\n--- modo streaming ---")
    recibidos = 0
    completo = False
    async for evento in manager.stream([SYSTEM, ChatMessage.user(PREGUNTA)]):
        match evento.type:
            case "token":
                recibidos += 1
                print(evento.text, end="", flush=True)
            case "error":
                print(f"\nerror controlado -> {evento.error}")
            case "done":
                print(f"\n\n[{recibidos} fragmentos | {evento.response.latency_ms} ms]")
                completo = True
    return completo


async def probar(provider: Provider) -> bool | None:
    """Prueba los dos modos. None si se salteo por falta de API key."""
    print(f"\n=== {provider.upper()} ===")
    if not os.getenv(API_KEY_ENV[provider]):
        print(f"salteado: no hay {API_KEY_ENV[provider]} en el entorno.")
        return None

    # El manager se cierra solo al salir del contexto.
    async with AsyncLLMManager.from_env(provider=provider) as manager:
        print(f"modelo: {manager.config.model}")
        normal = await modo_normal(manager)
        streaming = await modo_streaming(manager)
        return normal and streaming


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "provider",
        nargs="?",
        choices=["openai", "anthropic", "both"],
        help="Proveedor a probar. Por defecto, LLM_PROVIDER del .env.",
    )
    args = parser.parse_args()

    load_dotenv()
    if args.provider == "both":
        providers = list(Provider)
    elif args.provider:
        providers = [Provider(args.provider)]
    else:
        providers = [Provider(os.getenv("LLM_PROVIDER") or Provider.OPENAI)]

    resultados = [await probar(p) for p in providers]
    probados = [r for r in resultados if r is not None]

    if not probados:
        print("\nNingun proveedor probado: falta la API key. Copia .env.example a .env.")
        return 1
    if not all(probados):
        print("\nAlguna prueba fallo (ver los errores de arriba).")
        return 1
    print(f"\nListo: {len(probados)} proveedor(es) ok en modo normal y streaming.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
