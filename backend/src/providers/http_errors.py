"""Shared HTTP status and transport error mapping for provider adapters."""

from __future__ import annotations

import logging

import httpx

from src.providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class HTTPErrorMapper:
    """Map httpx failures to the provider exception hierarchy."""

    def __init__(self, provider: str, *, unavailable_message: str | None = None) -> None:
        self._provider = provider
        self._unavailable_message = unavailable_message or f"{provider} is unavailable."

    def map_request_error(self, exc: httpx.RequestError) -> ProviderUnavailableError:
        logger.error("%s request failed: %s", self._provider, type(exc).__name__)
        return ProviderUnavailableError(self._unavailable_message, provider=self._provider)

    def map_http_error(self, exc: httpx.HTTPStatusError) -> ProviderError:
        status = exc.response.status_code
        logger.error("%s HTTP error: status=%d", self._provider, status)

        if status == 404:
            return ModelNotFoundError(
                f"Requested {self._provider} model was not found.",
                provider=self._provider,
            )
        if status == 400:
            return InvalidRequestError(
                f"{self._provider} rejected the request.",
                provider=self._provider,
            )
        if status in (401, 403):
            return AuthenticationError(
                f"{self._provider} authentication failed.",
                provider=self._provider,
            )
        if status == 429:
            return RateLimitError(
                f"{self._provider} rate limit exceeded.",
                provider=self._provider,
            )
        if status >= 500:
            return ProviderUnavailableError(
                f"{self._provider} returned a server error.",
                provider=self._provider,
            )
        return ProviderError(
            f"{self._provider} request failed with HTTP {status}.",
            provider=self._provider,
        )

    def should_retry_status(self, status: int) -> bool:
        return status in (429, 502, 503, 504)

    def retry_after_seconds(self, response: httpx.Response) -> float | None:
        header = response.headers.get("Retry-After")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            return None
