"""Provider-layer exception hierarchy for normalized error handling."""

from __future__ import annotations


class ProviderError(Exception):
    """
    Base exception for all provider adapter failures.

    Catch this at the service or gateway boundary to map provider-specific
    errors into HTTP responses without leaking vendor details.
    """

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        self.provider = provider
        super().__init__(message)


class ProviderNotFoundError(ProviderError):
    """Raised when a requested provider identity is unknown or not registered."""


class ProviderUnavailableError(ProviderError):
    """Raised when the provider backend cannot be reached or is not ready."""


class AuthenticationError(ProviderError):
    """Raised when credentials are missing, invalid, or rejected by the provider."""


class ModelNotFoundError(ProviderError):
    """Raised when the requested model identifier is not offered by the provider."""


class RateLimitError(ProviderError):
    """Raised when the provider rejects a request due to rate limiting."""


class InvalidRequestError(ProviderError):
    """Raised when the provider rejects a request because the payload is invalid."""


class StreamingNotSupportedError(ProviderError):
    """Raised when streaming is requested but not supported by the provider or model."""
