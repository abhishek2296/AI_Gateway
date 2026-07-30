"""
Shared HTTP status and transport error mapping for provider adapters.

Every HTTP-based provider adapter (``OllamaProvider``, ``OpenAIProvider``,
``AnthropicProvider``, ``GeminiProvider``) makes calls through ``httpx`` and
needs to translate whatever comes back — a raw transport failure
(``httpx.RequestError``) or a non-2xx HTTP response (``httpx.HTTPStatusError``
raised via ``response.raise_for_status()``) — into one of the normalized
exceptions from ``providers/exceptions.py``.

``HTTPErrorMapper`` centralizes that translation logic in one place instead
of duplicating a status-code-to-exception ``if/elif`` chain in every adapter
module. The status-code mapping below follows the *de facto* convention
shared by OpenAI, Anthropic, and Google's APIs (and generally REST APIs at
large), so a single mapper implementation works for every current adapter.
Vendors with meaningfully different conventions can still opt out by
subclassing or by not using this mapper at all.
"""

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
    """
    Map httpx failures to the provider exception hierarchy.

    Each HTTP-based adapter constructs one ``HTTPErrorMapper`` instance
    (usually module-level or in ``__init__``) bound to its own provider name
    and a vendor-specific "unavailable" message, then routes every
    ``httpx.RequestError``/``httpx.HTTPStatusError`` it catches through
    :meth:`map_request_error`/:meth:`map_http_error` before re-raising.

    Attributes:
        _provider: The stable provider identifier used in log messages and
            in the ``provider`` field of every raised exception (e.g.
            ``"openai"``).
        _unavailable_message: The message used for
            :class:`~src.providers.exceptions.ProviderUnavailableError` when
            the underlying transport fails outright (as opposed to the
            provider returning a non-2xx response). Defaults to
            ``"{provider} is unavailable."`` when not supplied, but adapters
            typically pass a more specific, user-facing message (e.g.
            ``"Ollama is unavailable."``).

    Example:
        >>> mapper = HTTPErrorMapper("openai", unavailable_message="OpenAI is unavailable.")
        >>> mapper.should_retry_status(503)
        True
        >>> mapper.should_retry_status(404)
        False
    """

    def __init__(self, provider: str, *, unavailable_message: str | None = None) -> None:
        """
        Create a mapper bound to a single provider's identity and messaging.

        Args:
            provider: Stable provider identifier (e.g. ``"anthropic"``),
                recorded on every exception this mapper produces.
            unavailable_message: Optional human-readable message to use when
                the transport itself fails (connection refused, timeout,
                DNS failure, etc.). Defaults to a generic
                ``"{provider} is unavailable."`` if omitted.
        """
        self._provider = provider
        self._unavailable_message = unavailable_message or f"{provider} is unavailable."

    def map_request_error(self, exc: httpx.RequestError) -> ProviderUnavailableError:
        """
        Translate a low-level transport failure into a normalized exception.

        ``httpx.RequestError`` covers failures that happen *before* a response
        is received at all — DNS resolution failures, connection refused,
        connect/read timeouts, TLS errors, etc. None of these indicate
        anything about the request's validity, so they are always mapped to
        :class:`~src.providers.exceptions.ProviderUnavailableError` (a
        transient, likely-retryable condition) rather than any
        request-validation-flavored exception.

        Args:
            exc: The transport-level exception caught around the HTTP call.

        Returns:
            A :class:`~src.providers.exceptions.ProviderUnavailableError`
            carrying this mapper's configured unavailable message and
            provider name. Returned (not raised) so the caller controls the
            ``raise ... from exc`` chaining at the call site.

        Example:
            >>> import httpx
            >>> mapper = HTTPErrorMapper("ollama", unavailable_message="Ollama is unavailable.")
            >>> err = mapper.map_request_error(httpx.ConnectError("refused"))
            >>> str(err)
            'Ollama is unavailable.'
            >>> err.provider
            'ollama'
        """
        logger.error("%s request failed: %s", self._provider, type(exc).__name__)
        return ProviderUnavailableError(self._unavailable_message, provider=self._provider)

    def map_http_error(self, exc: httpx.HTTPStatusError) -> ProviderError:
        """
        Translate a non-2xx HTTP response into the matching provider exception.

        The status-code mapping reflects widely-shared REST API conventions
        used consistently by OpenAI, Anthropic, and Google Gemini:

        - ``404`` -> :class:`~src.providers.exceptions.ModelNotFoundError` —
          almost always because the requested model id doesn't exist for
          this vendor/endpoint.
        - ``400`` -> :class:`~src.providers.exceptions.InvalidRequestError` —
          the vendor rejected the payload shape/parameters.
        - ``401``/``403`` -> :class:`~src.providers.exceptions.AuthenticationError`
          — both "unauthenticated" and "forbidden" are surfaced identically
          to callers, since in practice both usually mean "your API key is
          missing, wrong, or lacks access" and callers rarely need to
          distinguish the two.
        - ``429`` -> :class:`~src.providers.exceptions.RateLimitError` — the
          one status this mapper also flags as retryable (see
          :meth:`should_retry_status`) and inspects for a ``Retry-After``
          hint (see :meth:`retry_after_seconds`).
        - ``>= 500`` -> :class:`~src.providers.exceptions.ProviderUnavailableError`
          — vendor-side outages are treated the same as transport failures,
          since from the caller's perspective both mean "the service isn't
          working right now, not that the request was wrong".
        - Anything else -> a generic
          :class:`~src.providers.exceptions.ProviderError` carrying the raw
          status code, so unexpected/future status codes still surface as a
          catchable, informative error instead of an unmapped exception.

        Args:
            exc: The ``httpx.HTTPStatusError`` raised by
                ``response.raise_for_status()`` after a non-2xx response.

        Returns:
            The most specific applicable ``ProviderError`` subclass for
            ``exc.response.status_code``, populated with this mapper's
            provider name.

        Example:
            >>> import httpx
            >>> mapper = HTTPErrorMapper("openai")
            >>> request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            >>> response = httpx.Response(429, request=request)
            >>> exc = httpx.HTTPStatusError("rate limited", request=request, response=response)
            >>> type(mapper.map_http_error(exc)).__name__
            'RateLimitError'
        """
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
        """
        Decide whether an HTTP status code represents a retryable failure.

        Used by :func:`~src.providers.retry.retry_async` to decide whether to
        back off and retry a request or fail immediately. The chosen set —
        ``429`` (rate limited), ``502``/``503``/``504`` (bad gateway, service
        unavailable, gateway timeout) — covers failures that are typically
        *transient*: the request itself was fine, but the vendor (or an
        intermediate proxy/load balancer in front of it) was temporarily
        unable to handle it. Deliberately excludes ``500`` (internal server
        error): a generic 500 does not reliably indicate a transient
        condition the way the gateway-specific 502/503/504 codes do, and
        retrying it risks repeating a request that failed for a persistent
        reason.

        Args:
            status: The HTTP status code from the failed response.

        Returns:
            ``True`` if the status code is one of ``429``, ``502``, ``503``,
            or ``504``; ``False`` otherwise.

        Example:
            >>> mapper = HTTPErrorMapper("openai")
            >>> mapper.should_retry_status(503)
            True
            >>> mapper.should_retry_status(500)
            False
        """
        return status in (429, 502, 503, 504)

    def retry_after_seconds(self, response: httpx.Response) -> float | None:
        """
        Read a vendor-supplied ``Retry-After`` hint from a response, if present.

        When a provider is explicit about how long to wait (most commonly on
        ``429`` rate-limit responses), honoring that hint is more accurate
        and more polite than blindly applying exponential backoff — it
        avoids retrying too early (getting rate-limited again) or waiting
        longer than necessary.

        Args:
            response: The HTTP response to inspect for a ``Retry-After``
                header.

        Returns:
            The parsed number of seconds to wait, or ``None`` if the header
            is absent or not parseable as a float. Note this implementation
            only handles the numeric-seconds form of ``Retry-After``; the
            HTTP spec also allows an HTTP-date value, which is not currently
            parsed and will therefore fall back to ``None`` (letting
            ``retry_async`` use exponential backoff instead).

        Example:
            >>> import httpx
            >>> mapper = HTTPErrorMapper("openai")
            >>> response = httpx.Response(429, headers={"Retry-After": "2.5"})
            >>> mapper.retry_after_seconds(response)
            2.5
            >>> response_no_header = httpx.Response(429)
            >>> mapper.retry_after_seconds(response_no_header) is None
            True
        """
        header = response.headers.get("Retry-After")
        if header is None:
            return None
        try:
            return float(header)
        except ValueError:
            # Retry-After may also be an HTTP-date per RFC 9110, which we
            # don't parse; falling back to None lets the caller use its own
            # exponential-backoff default instead of failing outright.
            return None
