"""`AsyncLLMManager`: elige el proveedor por configuracion y expone una sola API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from dotenv import load_dotenv

from anthropic_client import AnthropicClient
from base_client import BaseLLMClient
from openai_client import OpenAIClient
from schemas import (
    DEFAULT_MODELS,
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamEvent,
)

Prompt = str | Sequence[ChatMessage]

REGISTRY: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
}

ENV_MODEL_KEYS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_MODEL",
    Provider.ANTHROPIC: "ANTHROPIC_MODEL",
}


class AsyncLLMManager:
    """Fachada sobre los clientes concretos.

    El proveedor se resuelve una sola vez, al construir el manager, y desde
    ahi el codigo de aplicacion no sabe con cual esta hablando.
    """

    def __init__(self, config: ModelConfig | None = None, **overrides) -> None:
        self.config = config or ModelConfig(**overrides)
        try:
            client_cls = REGISTRY[self.config.provider]
        except KeyError:  # pragma: no cover - Provider ya valida el valor
            raise ValueError(f"Proveedor no soportado: {self.config.provider}") from None
        self._client: BaseLLMClient = client_cls(self.config)

    @classmethod
    def from_env(cls, **overrides) -> AsyncLLMManager:
        """Construye el manager leyendo el `.env` (ver `.env.example`)."""
        load_dotenv()
        provider = Provider(
            overrides.pop("provider", None) or os.getenv("LLM_PROVIDER") or Provider.OPENAI
        )
        model = overrides.pop("model", None) or os.getenv(ENV_MODEL_KEYS[provider])
        settings: dict[str, object] = {
            "provider": provider,
            "model": model or DEFAULT_MODELS[provider],
        }
        for key, cast in (
            ("LLM_TEMPERATURE", float),
            ("LLM_MAX_TOKENS", int),
            ("LLM_TIMEOUT", float),
            ("LLM_MAX_RETRIES", int),
            ("LLM_EFFORT", str),
        ):
            raw = os.getenv(key)
            if raw:
                settings[key.removeprefix("LLM_").lower()] = cast(raw)
        settings.update(overrides)
        return cls(ModelConfig(**settings))

    # ---------------------------------------------------------------- API

    @property
    def provider(self) -> Provider:
        return self.config.provider

    async def generate(self, prompt: Prompt) -> ModelResponse:
        """Respuesta completa. Nunca lanza por errores de API."""
        return await self._client.generate(_as_messages(prompt))

    def stream(self, prompt: Prompt) -> AsyncIterator[StreamEvent]:
        """Generador asincronico de tokens (ver `StreamEvent`)."""
        return self._client.stream(_as_messages(prompt))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncLLMManager:
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()


def _as_messages(prompt: Prompt) -> list[ChatMessage]:
    """Acepta un string suelto o una lista de `ChatMessage` ya armada."""
    if isinstance(prompt, str):
        return [ChatMessage.user(prompt)]
    return list(prompt)
