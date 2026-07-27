"""Shared HTTP client lifecycle for REST-based provider adapters."""

from __future__ import annotations

import logging
from typing import Self

import httpx

logger = logging.getLogger(__name__)


def require_non_empty_string(value: str, field_name: str) -> str:
    """Validate required string constructor arguments."""
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


class HTTPProviderMixin:
    """
    Reusable ``httpx.AsyncClient`` ownership and async context manager support.

    Concrete providers call :meth:`_init_http_client` from ``__init__`` and inherit
    :meth:`close`, :meth:`__aenter__`, and :meth:`__aexit__` without duplication.
    """

    _base_url: str
    _timeout: float
    _owns_client: bool
    _client: httpx.AsyncClient

    def _init_http_client(
        self,
        *,
        base_url: str,
        timeout: float,
        http_client: httpx.AsyncClient | None,
        provider_label: str,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )
        logger.info(
            "Initialized %s (base_url=%s, owns_client=%s)",
            provider_label,
            self._base_url,
            self._owns_client,
        )

    async def close(self) -> None:
        """Close the internally created HTTP client, if any."""
        if self._owns_client:
            await self._client.aclose()
            logger.info("Closed %s HTTP client", self.__class__.__name__)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
