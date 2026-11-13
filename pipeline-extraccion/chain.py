"""Pipeline LCEL de extracción de entidades técnicas.

Flujo:  texto -> prompt -> modelo con salida estructurada -> validación -> objeto Pydantic
Con reintento automático si la salida viene mal formada, incompleta o falla la red.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from schemas import ExtraccionTecnica

load_dotenv()  # carga OPENAI_API_KEY / ANTHROPIC_API_KEY desde .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# --------------------------------------------------------------------------
# 1. Modelo: se elige el proveedor por configuración (idea del Módulo 1)
# --------------------------------------------------------------------------
def get_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            max_tokens=500,
            max_retries=0,  # los reintentos los maneja with_retry, no el SDK
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            temperature=0,
            max_tokens=500,
            max_retries=0,
        )
    raise ValueError(f"LLM_PROVIDER no soportado: {provider!r} (usá 'openai' o 'anthropic')")


# --------------------------------------------------------------------------
# 2. Prompt modular (sin f-strings: las variables las maneja LangChain)
# --------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un ingeniero de software senior especializado en analizar logs "
            "y descripciones de arquitectura. Extraé la información técnica del "
            "texto del usuario siguiendo estas reglas:\n"
            "- Incluí solo tecnologías mencionadas explícitamente; no inventes.\n"
            "- Evaluá la criticidad según el impacto descrito.\n"
            "- Escribí el resumen en español, en una o dos oraciones.",
        ),
        ("human", "{texto}"),
    ]
)


# --------------------------------------------------------------------------
# 3. Validación extra: finish_reason y errores de parseo
# --------------------------------------------------------------------------
class SalidaIncompletaError(Exception):
    """El modelo cortó la respuesta por límite de tokens."""


def validar_salida(resultado: dict) -> ExtraccionTecnica:
    """Recibe {'raw', 'parsed', 'parsing_error'} y devuelve el objeto validado.

    Si algo está mal lanza una excepción, y eso dispara el reintento.
    """
    raw = resultado["raw"]
    metadata = raw.response_metadata or {}

    # OpenAI usa finish_reason="length"; Anthropic usa stop_reason="max_tokens"
    motivo_fin = metadata.get("finish_reason") or metadata.get("stop_reason")
    if motivo_fin in ("length", "max_tokens"):
        logger.warning("Salida incompleta (finish_reason=%s). Se reintentará.", motivo_fin)
        raise SalidaIncompletaError(f"Respuesta cortada: {motivo_fin}")

    if resultado["parsing_error"] is not None:
        logger.warning(
            "Validación fallida: %s. Se reintentará.",
            type(resultado["parsing_error"]).__name__,
        )
        raise resultado["parsing_error"]

    logger.info("Validación OK con Pydantic")
    return resultado["parsed"]


# --------------------------------------------------------------------------
# 4. Cadena LCEL con reintentos
# --------------------------------------------------------------------------
# include_raw=True devuelve también la respuesta cruda, para leer el finish_reason
modelo_estructurado = get_model().with_structured_output(ExtraccionTecnica, include_raw=True)

# Errores que justifican reintentar: salida mal formada/incompleta.
# (los errores de red del SDK se agregan abajo si están disponibles)
ERRORES_REINTENTABLES: tuple[type[BaseException], ...] = (
    ValidationError,
    OutputParserException,
    SalidaIncompletaError,
)
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    ERRORES_REINTENTABLES += (APIConnectionError, APITimeoutError, RateLimitError)
except ImportError:
    pass

extractor = (modelo_estructurado | RunnableLambda(validar_salida)).with_retry(
    retry_if_exception_type=ERRORES_REINTENTABLES,
    stop_after_attempt=3,  # 1 intento + 2 reintentos
    wait_exponential_jitter=True,  # backoff exponencial entre intentos
)

chain = prompt | extractor


# --------------------------------------------------------------------------
# 5. Función pública asíncrona
# --------------------------------------------------------------------------
async def process_text(text: str) -> ExtraccionTecnica:
    """Procesa un texto libre y devuelve un objeto ExtraccionTecnica validado."""
    logger.info("Procesando texto (%d caracteres)", len(text))
    try:
        resultado = await chain.ainvoke({"texto": text})
    except Exception as e:
        logger.error("Falló tras todos los reintentos: %s - %s", type(e).__name__, e)
        raise
    logger.info(
        "Extracción exitosa: %d tecnologías, criticidad=%s",
        len(resultado.tecnologias),
        resultado.nivel_de_criticidad.value,
    )
    return resultado
