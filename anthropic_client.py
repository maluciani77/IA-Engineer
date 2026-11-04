"""Cliente asincronico de Anthropic (`AsyncAnthropic`)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

import anthropic
from anthropic import NOT_GIVEN, AsyncAnthropic

from base_client import BaseLLMClient
from schemas import ChatMessage, ErrorKind, ModelConfig, ModelResponse, Provider, Usage


class AnthropicClient(BaseLLMClient):
    """Implementa la interfaz comun sobre la Messages API.

    Diferencias con OpenAI que absorbe esta clase:

    - el system prompt viaja en su propio parametro, no como mensaje;
    - la API ya no acepta `temperature` ni `top_p`: el equivalente es
      `output_config.effort` (`ModelConfig.effort`).
    """

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Falta ANTHROPIC_API_KEY (ver .env.example).")
        # max_retries=0: los reintentos los maneja BaseLLMClient.
        self._client = AsyncAnthropic(api_key=key, timeout=config.timeout, max_retries=0)

    @property
    def provider(self) -> Provider:
        return Provider.ANTHROPIC

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------ interno

    def _payload(self, messages: Sequence[ChatMessage]) -> dict:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]
        return {
            "model": self.config.model,
            "messages": turns,
            "system": system or NOT_GIVEN,
            "max_tokens": self.config.max_tokens,
            "output_config": (
                {"effort": self.config.effort} if self.config.effort else NOT_GIVEN
            ),
        }

    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        message = await self._client.messages.create(**self._payload(messages))
        text = "".join(block.text for block in message.content if block.type == "text")
        return ModelResponse(
            provider=self.provider,
            model=message.model,
            content=text,
            finish_reason=message.stop_reason,
            usage=Usage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
            ),
        )

    async def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._payload(messages)) as stream:
            async for event in stream:
                # El helper emite muchos tipos de evento; solo nos interesa el texto.
                if event.type == "text":
                    yield event.text

    def _classify(self, exc: Exception) -> tuple[ErrorKind, int | None]:
        status = getattr(exc, "status_code", None)
        match exc:
            case anthropic.AuthenticationError() | anthropic.PermissionDeniedError():
                return ErrorKind.AUTH, status
            case anthropic.RateLimitError():
                return ErrorKind.RATE_LIMIT, status
            case anthropic.APITimeoutError():
                return ErrorKind.TIMEOUT, status
            case anthropic.APIConnectionError():
                return ErrorKind.CONNECTION, status
            case anthropic.InternalServerError():
                return ErrorKind.SERVER, status
            case anthropic.BadRequestError() | anthropic.NotFoundError():
                return ErrorKind.BAD_REQUEST, status
            case anthropic.APIStatusError():
                kind = ErrorKind.SERVER if (status or 0) >= 500 else ErrorKind.BAD_REQUEST
                return kind, status
            case _:
                return ErrorKind.UNKNOWN, status
