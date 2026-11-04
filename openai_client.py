"""Cliente asincronico de OpenAI (`AsyncOpenAI`)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

import openai
from openai import AsyncOpenAI

from base_client import BaseLLMClient
from schemas import ChatMessage, ErrorKind, ModelConfig, ModelResponse, Provider, Usage


class OpenAIClient(BaseLLMClient):
    """Implementa la interfaz comun sobre `chat.completions`."""

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Falta OPENAI_API_KEY (ver .env.example).")
        # max_retries=0: los reintentos los maneja BaseLLMClient.
        self._client = AsyncOpenAI(
            api_key=key, timeout=config.timeout, max_retries=0
        )

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------ interno

    def _payload(self, messages: Sequence[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        completion = await self._client.chat.completions.create(**self._payload(messages))
        choice = completion.choices[0]
        usage = None
        if completion.usage is not None:
            usage = Usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
        return ModelResponse(
            provider=self.provider,
            model=completion.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    async def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._payload(messages), stream=True
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _classify(self, exc: Exception) -> tuple[ErrorKind, int | None]:
        status = getattr(exc, "status_code", None)
        match exc:
            case openai.AuthenticationError() | openai.PermissionDeniedError():
                return ErrorKind.AUTH, status
            case openai.RateLimitError():
                return ErrorKind.RATE_LIMIT, status
            case openai.APITimeoutError():
                return ErrorKind.TIMEOUT, status
            case openai.APIConnectionError():
                return ErrorKind.CONNECTION, status
            case openai.InternalServerError():
                return ErrorKind.SERVER, status
            case openai.BadRequestError() | openai.NotFoundError():
                return ErrorKind.BAD_REQUEST, status
            case openai.APIStatusError():
                kind = ErrorKind.SERVER if (status or 0) >= 500 else ErrorKind.BAD_REQUEST
                return kind, status
            case _:
                return ErrorKind.UNKNOWN, status
