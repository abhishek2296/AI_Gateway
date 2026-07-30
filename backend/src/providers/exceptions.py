"""
Provider-layer exception hierarchy for normalized error handling.

Every concrete provider adapter (``OllamaProvider``, ``OpenAIProvider``,
``AnthropicProvider``, ``GeminiProvider``, ...) talks to a different vendor
API with its own status codes, error payloads, and quirks. Rather than let
those vendor-specific exceptions (``httpx.HTTPStatusError``, SDK-specific
error types, etc.) leak up into the gateway's service/route layers, every
adapter is required to catch vendor errors at the transport boundary and
re-raise one of the exceptions defined here instead (see
``http_errors.HTTPErrorMapper`` for the HTTP status -> exception mapping used
by the HTTP-based adapters).

This gives the rest of the codebase (services, routes, the global FastAPI
exception handler) a single, small, provider-agnostic vocabulary to catch and
react to, regardless of which vendor actually served the request.
"""

from __future__ import annotations


class ProviderError(Exception):
    """
    Base exception for all provider adapter failures.

    Catch this at the service or gateway boundary to map provider-specific
    errors into HTTP responses without leaking vendor details. All other
    exceptions in this module inherit from ``ProviderError``, so a single
    ``except ProviderError:`` clause is sufficient to handle *any* provider
    failure generically; catch a specific subclass instead when the caller
    needs to react differently depending on the failure mode (e.g. retry on
    ``RateLimitError`` but fail fast on ``AuthenticationError``).

    Attributes:
        provider: The stable identifier of the provider that raised this
            error (e.g. ``"openai"``, ``"ollama"``), or ``None`` if the error
            was not raised in the context of a specific provider instance.
            Useful for building precise log messages and error responses
            that name the offending backend.

    Example:
        >>> try:
        ...     raise ProviderError("boom", provider="openai")
        ... except ProviderError as exc:
        ...     print(f"{exc.provider}: {exc}")
        openai: boom
    """

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        """
        Create the exception with a human-readable message and provider name.

        Args:
            message: A human-readable explanation of what went wrong. Becomes
                the exception's string representation (``str(exc)``).
            provider: Optional stable provider identifier (e.g. ``"gemini"``).
                Keyword-only so call sites always read clearly, e.g.
                ``ProviderError("timed out", provider="gemini")``.
        """
        self.provider = provider
        super().__init__(message)


class ProviderNotFoundError(ProviderError):
    """
    Raised when a requested provider identity is unknown or not registered.

    Raised by ``ProviderRegistry.get``/``unregister`` (see ``registry.py``)
    when the caller asks for a provider name that has never been registered
    — for example, a typo in configuration (``"opneai"``) or a provider that
    has not shipped yet.

    Example:
        >>> from src.providers.registry import ProviderRegistry
        >>> registry = ProviderRegistry()
        >>> registry.get("does-not-exist")
        Traceback (most recent call last):
            ...
        ProviderNotFoundError: Provider 'does-not-exist' is not registered.
    """


class ProviderUnavailableError(ProviderError):
    """
    Raised when the provider backend cannot be reached or is not ready.

    Covers both network-level failures (DNS errors, connection refused,
    timeouts — see ``HTTPErrorMapper.map_request_error``) and vendor-side
    outages signalled via ``5xx`` HTTP responses (see
    ``HTTPErrorMapper.map_http_error``). This is the exception a caller
    should treat as "safe/likely worth retrying or falling back to another
    provider", as opposed to e.g. ``AuthenticationError`` which will not be
    fixed by retrying.

    Example:
        >>> raise ProviderUnavailableError(
        ...     "Ollama is unavailable.", provider="ollama"
        ... )
        Traceback (most recent call last):
            ...
        ProviderUnavailableError: Ollama is unavailable.
    """


class AuthenticationError(ProviderError):
    """
    Raised when credentials are missing, invalid, or rejected by the provider.

    Maps from HTTP ``401``/``403`` responses (see
    ``HTTPErrorMapper.map_http_error``). Unlike ``ProviderUnavailableError``,
    this is generally *not* worth retrying — the request will keep failing
    until the API key/credentials are fixed.

    Example:
        >>> raise AuthenticationError(
        ...     "openai authentication failed.", provider="openai"
        ... )
        Traceback (most recent call last):
            ...
        AuthenticationError: openai authentication failed.
    """


class ModelNotFoundError(ProviderError):
    """
    Raised when the requested model identifier is not offered by the provider.

    Maps from HTTP ``404`` responses (see ``HTTPErrorMapper.map_http_error``)
    and can also be raised by higher-level validation (e.g.
    ``BaseProvider.validate_model``) once a provider's model catalog has
    been fetched and the requested model isn't in it.

    Example:
        >>> raise ModelNotFoundError(
        ...     "Requested openai model was not found.", provider="openai"
        ... )
        Traceback (most recent call last):
            ...
        ModelNotFoundError: Requested openai model was not found.
    """


class RateLimitError(ProviderError):
    """
    Raised when the provider rejects a request due to rate limiting.

    Maps from HTTP ``429`` responses. This is the one status code that both
    ``HTTPErrorMapper.should_retry_status`` flags as retryable *and* that
    ``HTTPErrorMapper.retry_after_seconds`` inspects a ``Retry-After`` header
    for, so ``retry_async`` (see ``retry.py``) can back off for exactly as
    long as the vendor asked rather than guessing.

    Example:
        >>> raise RateLimitError(
        ...     "openai rate limit exceeded.", provider="openai"
        ... )
        Traceback (most recent call last):
            ...
        RateLimitError: openai rate limit exceeded.
    """


class InvalidRequestError(ProviderError):
    """
    Raised when the provider rejects a request because the payload is invalid.

    Maps from HTTP ``400`` responses. Distinct from a local validation error
    (which would fail before any HTTP call is made) — this specifically means
    the *vendor* rejected a request the gateway believed was well-formed,
    e.g. an unsupported parameter combination, invalid model name for the
    endpoint, or malformed message content.

    Example:
        >>> raise InvalidRequestError(
        ...     "openai rejected the request.", provider="openai"
        ... )
        Traceback (most recent call last):
            ...
        InvalidRequestError: openai rejected the request.
    """


class StreamingNotSupportedError(ProviderError):
    """
    Raised when streaming is requested but not supported by the provider or model.

    Declared on ``BaseProvider.stream_chat`` as a possible failure mode for
    providers/models that only support non-streaming completions; not every
    concrete adapter in this codebase currently raises it (all four
    implemented adapters support streaming), but adapters for future
    providers/models without streaming support should raise it here instead
    of silently falling back to a blocking call.

    Example:
        >>> raise StreamingNotSupportedError(
        ...     "azure-openai does not support streaming for this model.",
        ...     provider="azure-openai",
        ... )
        Traceback (most recent call last):
            ...
        StreamingNotSupportedError: azure-openai does not support streaming for this model.
    """


class UnsupportedCapabilityError(ProviderError):
    """
    Raised when a provider does not support a requested capability.

    Used for capabilities that are fundamentally unavailable from a vendor's
    API rather than being a transient failure — for example,
    ``AnthropicProvider.embeddings`` raises this because Anthropic's Messages
    API has no embeddings endpoint, and the default
    ``BaseProvider.estimate_tokens`` implementation raises this for any
    provider that hasn't implemented token estimation.

    Example:
        >>> raise UnsupportedCapabilityError(
        ...     "Anthropic does not expose an embeddings API.",
        ...     provider="anthropic",
        ... )
        Traceback (most recent call last):
            ...
        UnsupportedCapabilityError: Anthropic does not expose an embeddings API.
    """
