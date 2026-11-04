"""Esquemas Pydantic compartidos por todos los clientes LLM.

Toda la informacion que entra y sale del cliente pasa por estos modelos, asi
que nunca se manipulan diccionarios anidados sueltos.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provider(StrEnum):
    """Proveedores soportados."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


Role = Literal["system", "user", "assistant"]

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-5-5",
}


class ChatMessage(BaseModel):
    """Un turno de conversacion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str = Field(min_length=1)

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role="assistant", content=content)


class ModelConfig(BaseModel):
    """Parametros de invocacion validados.

    Un mismo objeto sirve para los dos proveedores, pero no todos los campos
    aplican a ambos:

    - `temperature` y `top_p` los usa OpenAI. La Messages API de Anthropic ya
      no los expone, asi que su cliente los ignora.
    - `effort` lo usa Anthropic (`output_config.effort`). OpenAI lo ignora.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    provider: Provider = Provider.OPENAI
    model: str = Field(default="", min_length=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=32_000)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    effort: Effort | None = None
    timeout: float = Field(default=30.0, gt=0.0, description="Segundos por intento.")
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay: float = Field(default=0.5, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def _fill_default_model(cls, data: object) -> object:
        """Si no se indica modelo, usa el default del proveedor elegido."""
        if isinstance(data, dict) and not data.get("model"):
            provider = Provider(data.get("provider") or Provider.OPENAI)
            return {**data, "model": DEFAULT_MODELS[provider]}
        return data


class ErrorKind(StrEnum):
    """Categorias de error normalizadas entre proveedores."""

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER = "server"
    BAD_REQUEST = "bad_request"
    UNKNOWN = "unknown"


RETRYABLE_ERRORS: frozenset[ErrorKind] = frozenset(
    {ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.CONNECTION, ErrorKind.SERVER}
)


class ErrorInfo(BaseModel):
    """Error controlado: se devuelve en lugar de propagar la excepcion."""

    kind: ErrorKind
    message: str
    provider: Provider
    status_code: int | None = None
    attempts: int = Field(default=1, ge=1)

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_ERRORS

    def __str__(self) -> str:
        code = f" [{self.status_code}]" if self.status_code else ""
        return f"{self.provider}/{self.kind}{code}: {self.message}"


class Usage(BaseModel):
    """Tokens consumidos por la llamada."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelResponse(BaseModel):
    """Respuesta unificada. Si `error` viene cargado, `ok` es False."""

    provider: Provider
    model: str
    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: ErrorInfo | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class StreamEvent(BaseModel):
    """Evento del generador asincronico.

    - `token`: fragmento de texto recien llegado (`text`).
    - `done`: fin del stream, con la respuesta completa en `response`.
    - `error`: fallo controlado; el generador termina despues de emitirlo.
    """

    type: Literal["token", "done", "error"]
    text: str = ""
    error: ErrorInfo | None = None
    response: ModelResponse | None = None
