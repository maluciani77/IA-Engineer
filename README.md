# Unified Async LLM Client

Cliente unificado y asincronico para OpenAI y Anthropic. Los dos proveedores
se usan con la misma interfaz, las llamadas no bloquean el event loop, el
streaming llega token por token y ningun error de API rompe el programa.

Pre-entrega 1 del modulo de AI Engineering.

## Como correrlo

Requiere Python 3.12 o superior.

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env    # y completa tu API key
```

```bash
python main.py both
```

`main.py` hace la misma pregunta ("¿Que es la entropia?") en modo normal y en
modo streaming. Acepta `openai`, `anthropic` o `both`; sin argumento usa
`LLM_PROVIDER` del `.env`. Si falta la key de un proveedor, lo saltea en lugar
de fallar.

## Variables de entorno

Todas se leen del `.env` (ver `.env.example`). Solo las dos primeras son
necesarias.

| Variable | Default | Para que sirve |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | Proveedor por defecto: `openai` o `anthropic`. |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Key del proveedor que vayas a usar. |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` | `gpt-4o-mini` / `claude-sonnet-5-5` | Modelo a invocar. |
| `LLM_TEMPERATURE` | `0.7` | Validada entre 0 y 2. Solo OpenAI. |
| `LLM_EFFORT` | — | `low`…`max`. Solo Anthropic (ver nota abajo). |
| `LLM_MAX_TOKENS` | `512` | Tope de tokens de salida. |
| `LLM_TIMEOUT` | `30` | Segundos por intento. |
| `LLM_MAX_RETRIES` | `3` | Reintentos ante errores transitorios. |

El `.env` esta en `.gitignore`; al repo solo sube `.env.example`.

## Estructura

| Archivo | Contenido |
| --- | --- |
| `schemas.py` | Modelos Pydantic: `ChatMessage`, `ModelConfig`, `ModelResponse`, `StreamEvent`, `ErrorInfo`. |
| `base_client.py` | `BaseLLMClient` abstracta: `async generate()`, `stream()`, reintentos y manejo de errores. |
| `openai_client.py` | `OpenAIClient` sobre `AsyncOpenAI`. |
| `anthropic_client.py` | `AnthropicClient` sobre `AsyncAnthropic`. |
| `manager.py` | `AsyncLLMManager`: elige el proveedor por configuracion. |
| `main.py` | Script de prueba de los dos modos. |

## Uso desde codigo

```python
import asyncio
from manager import AsyncLLMManager

async def main():
    async with AsyncLLMManager.from_env() as llm:
        # Modo normal
        r = await llm.generate("¿Que es la entropia?")
        print(r.content if r.ok else r.error)

        # Modo streaming
        async for evento in llm.stream("¿Que es la entropia?"):
            if evento.type == "token":
                print(evento.text, end="", flush=True)
            elif evento.type == "error":
                print(f"\nfallo: {evento.error}")

asyncio.run(main())
```

Para fijar parametros a mano, sin `.env`:

```python
from manager import AsyncLLMManager
from schemas import ModelConfig, Provider

llm = AsyncLLMManager(ModelConfig(provider=Provider.ANTHROPIC, max_tokens=256))
```

## Decisiones de diseno

**Asincronia real.** Se usan `AsyncOpenAI` y `AsyncAnthropic` con `await`.
Usar los clientes sincronicos dentro de una funcion `async` bloquearia el
event loop entero mientras el modelo responde.

**Los errores se devuelven, no se lanzan.** `generate()` siempre retorna un
`ModelResponse`: si algo fallo, `response.ok` es `False` y `response.error`
trae un `ErrorInfo` con la categoria (`auth`, `rate_limit`, `timeout`,
`connection`, `server`, `bad_request`, `unknown`), el status HTTP y cuantos
intentos se hicieron. Asi un error de key invalida o de cuota no corta el loop
principal. La unica excepcion que se propaga es `asyncio.CancelledError`,
porque tragarla rompe la cancelacion de tareas.

**Reintentos propios.** Los SDKs se instancian con `max_retries=0` y el
reintento lo hace `BaseLLMClient` con backoff exponencial y jitter, solo para
errores transitorios (`rate_limit`, `timeout`, `connection`, `server`). Un
`401` no se reintenta: no va a mejorar.

En streaming solo se reintenta si el fallo ocurre **antes** del primer token;
despues de haber emitido texto, repetir la llamada duplicaria la salida.

**Streaming con eventos tipados.** `stream()` es un generador asincronico que
emite `StreamEvent`:

- `token`: fragmento de texto en `.text`;
- `done`: fin, con la respuesta completa en `.response`;
- `error`: fallo controlado en `.error`, y el generador termina.

Devolver eventos en lugar de `str` sueltos permite reportar un error a mitad
del stream sin lanzar una excepcion al consumidor.

**Validacion antes de la red.** `ModelConfig` usa `extra="forbid"`, asi que un
typo como `temperatura=0.5` falla al construir el objeto en vez de ser
ignorado en silencio. `temperature` esta acotada a 0–2, `max_tokens` a 1–32000
y `top_p` a 0–1.

### Diferencias entre proveedores que absorbe el cliente

- El system prompt va como mensaje en OpenAI y como parametro aparte en
  Anthropic; `AnthropicClient` lo separa solo.
- Los nombres de tokens difieren (`prompt_tokens`/`completion_tokens` vs
  `input_tokens`/`output_tokens`) y se normalizan en `Usage`.
- `temperature` y `top_p` ya no existen en la Messages API de Anthropic: su
  equivalente es `output_config.effort`. `ModelConfig` expone los tres campos
  y cada cliente usa el que le corresponde.

## Verificacion

Probado contra un servidor HTTP local que imita ambas APIs (respuestas
normales y SSE), sin consumir cuota: los dos proveedores responden en modo
normal y streaming, un `429` se reintenta y termina bien, y un `401` devuelve
`ErrorInfo(kind=auth)` en lugar de romper.
