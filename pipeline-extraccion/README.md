# Pipeline de extracción de entidades técnicas

Pre-entrega 2 · AI Engineering (Coderhouse)

Recibe un texto libre (un log de error o una descripción de arquitectura) y devuelve un objeto **validado con Pydantic** con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico.

## Cómo funciona

```
texto ──► ChatPromptTemplate ──► LLM.with_structured_output(ExtraccionTecnica) ──► validación ──► objeto Pydantic
                                    └───────────── with_retry (hasta 3 intentos) ─────────────┘
```

| Archivo | Qué contiene |
|---|---|
| `schemas.py` | Modelo Pydantic `ExtraccionTecnica` y el enum `NivelCriticidad` |
| `chain.py` | Prompt, cadena LCEL, reintentos, logs y la función async `process_text()` |
| `main.py` | Mini-script de prueba con 3 textos (incluye una prueba de estrés) |

### Decisiones de diseño

- **Contrato estricto:** `tecnologias` no puede estar vacía (`min_length=1`), `nivel_de_criticidad` es un enum (`baja`, `media`, `alta`) y `resumen_tecnico` exige un mínimo de 10 caracteres.
- **Salida estructurada:** se usa `with_structured_output(..., include_raw=True)` para recibir el objeto parseado y además la respuesta cruda del modelo.
- **Control de `finish_reason`:** si el modelo cortó la respuesta por límite de tokens (`length` en OpenAI, `max_tokens` en Anthropic), se considera incompleta y se reintenta en lugar de intentar transformarla.
- **Resiliencia:** `.with_retry()` reintenta hasta 3 veces, con backoff exponencial, ante JSON mal formado o incompleto (`ValidationError`, `OutputParserException`) y errores de red o rate limit. El reintento interno del SDK está desactivado (`max_retries=0`) para que no se multipliquen los intentos.
- **Prompt modular:** `ChatPromptTemplate` con la variable `{texto}`, sin f-strings.
- **Proveedor configurable:** OpenAI o Anthropic según la variable `LLM_PROVIDER`, reutilizando la idea de abstracción del Módulo 1.
- **Logs:** cada paso (inicio, validación OK, validación fallida, reintento, error final) queda registrado con `logging`.

## Cómo correrlo

Requiere Python 3.12+.

```bash
# 1. Entorno virtual
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# 2. Dependencias
pip install -r requirements.txt

# 3. Variables de entorno
cp .env.example .env             # y completá tu API key

# 4. Ejecutar la prueba
python main.py
```

### Variables de entorno

| Variable | Descripción |
|---|---|
| `LLM_PROVIDER` | `openai` (default) o `anthropic` |
| `OPENAI_API_KEY` | Clave de OpenAI |
| `OPENAI_MODEL` | Modelo de OpenAI (default `gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Clave de Anthropic |
| `ANTHROPIC_MODEL` | Modelo de Anthropic |

## Ejemplo de salida

Entrada:

> La API en FastAPI devuelve 503 desde hace 20 minutos. Redis alcanzó el límite de memoria y las consultas a PostgreSQL superan el pool de 100 conexiones. Los usuarios no pueden hacer checkout.

Salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API en FastAPI devuelve 503 porque Redis se quedó sin memoria y PostgreSQL agotó su pool de conexiones, bloqueando el checkout."
}
```

Logs:

```
14:03:25 | INFO    | Procesando texto (231 caracteres)
14:03:27 | INFO    | Validación OK con Pydantic
14:03:27 | INFO    | Extracción exitosa: 3 tecnologías, criticidad=alta
```

## Prueba de estrés

El tercer texto de `main.py` no menciona ninguna tecnología ("Hoy almorcé milanesas con puré..."). Como `tecnologias` exige al menos un elemento, Pydantic rechaza una lista vacía: el pipeline registra la validación fallida, reintenta y, si el modelo no puede cumplir el contrato, termina con un error controlado en lugar de devolver datos inválidos:

```
INFO    | Procesando texto (66 caracteres)
WARNING | Validación fallida: ValidationError. Se reintentará.
WARNING | Validación fallida: ValidationError. Se reintentará.
WARNING | Validación fallida: ValidationError. Se reintentará.
ERROR   | Falló tras todos los reintentos: ValidationError - ...
No se pudo extraer: ValidationError
```

(El modelo también podría devolver algo forzado; en ese caso conviene revisar la salida. El comportamiento exacto puede variar entre ejecuciones.)
