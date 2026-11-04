"""Clase base abstracta para los clientes LLM asincronicos.

Centraliza lo que no depende del proveedor: reintentos con backoff
exponencial, traduccion de excepciones a `ErrorInfo` y medicion de latencia.
Cada proveedor solo implementa `_generate_once`, `_stream_once` y `_classify`.
"""

from __future__ import annotations

import abc
import asyncio
import random
import time
from collections.abc import AsyncIterator, Sequence

from schemas import (
    ChatMessage,
    ErrorInfo,
    ErrorKind,
    ModelConfig,
    ModelResponse,
    Provider,
    StreamEvent,
)


class BaseLLMClient(abc.ABC):
    """Interfaz comun: los dos proveedores son intercambiables."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # ---------------------------------------------------------------- API

    @property
    @abc.abstractmethod
    def provider(self) -> Provider:
        """Proveedor que implementa esta subclase."""

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Una sola respuesta completa, con reintentos y error controlado."""
        started = time.perf_counter()
        attempt = 0

        while True:
            attempt += 1
            try:
                response = await self._generate_once(messages)
            except Exception as exc:  # noqa: BLE001 - se normaliza abajo
                info = self._to_error_info(exc, attempt)
                if info.retryable and attempt <= self.config.max_retries:
                    await self._sleep_backoff(attempt)
                    continue
                return ModelResponse(
                    provider=self.provider,
                    model=self.config.model,
                    latency_ms=self._elapsed_ms(started),
                    error=info,
                )

            response.latency_ms = self._elapsed_ms(started)
            return response

    async def stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[StreamEvent]:
        """Generador asincronico de tokens.

        Reintenta solo si el fallo ocurre antes del primer token: una vez que
        el consumidor vio texto, repetir la llamada duplicaria la salida.
        """
        started = time.perf_counter()
        attempt = 0

        while True:
            attempt += 1
            pieces: list[str] = []
            try:
                async for delta in self._stream_once(messages):
                    if not delta:
                        continue
                    pieces.append(delta)
                    yield StreamEvent(type="token", text=delta)
            except Exception as exc:  # noqa: BLE001 - se normaliza abajo
                info = self._to_error_info(exc, attempt)
                if (
                    not pieces
                    and info.retryable
                    and attempt <= self.config.max_retries
                ):
                    await self._sleep_backoff(attempt)
                    continue
                yield StreamEvent(type="error", error=info)
                return

            yield StreamEvent(
                type="done",
                response=ModelResponse(
                    provider=self.provider,
                    model=self.config.model,
                    content="".join(pieces),
                    finish_reason="stop",
                    latency_ms=self._elapsed_ms(started),
                ),
            )
            return

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Cierra el cliente HTTP subyacente."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()

    # ----------------------------------------------- a implementar abajo

    @abc.abstractmethod
    async def _generate_once(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Una llamada sin reintentos al SDK del proveedor."""

    @abc.abstractmethod
    def _stream_once(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Un stream sin reintentos; emite los fragmentos de texto crudos."""

    @abc.abstractmethod
    def _classify(self, exc: Exception) -> tuple[ErrorKind, int | None]:
        """Traduce una excepcion del SDK a (categoria, status_code)."""

    # ------------------------------------------------------------ helpers

    def _to_error_info(self, exc: Exception, attempt: int) -> ErrorInfo:
        if isinstance(exc, asyncio.CancelledError):  # nunca lo tragamos
            raise exc
        kind, status = self._classify(exc)
        message = str(exc) or exc.__class__.__name__
        return ErrorInfo(
            kind=kind,
            message=message,
            provider=self.provider,
            status_code=status,
            attempts=attempt,
        )

    async def _sleep_backoff(self, attempt: int) -> None:
        """Backoff exponencial con jitter para no sincronizar reintentos."""
        delay = self.config.retry_base_delay * (2 ** (attempt - 1))
        await asyncio.sleep(min(delay, 8.0) * (0.5 + random.random() / 2))

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)

