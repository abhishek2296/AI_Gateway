"""
Shared HTTP client lifecycle for REST-based provider adapters.

Every REST-based provider adapter (``OllamaProvider``, ``OpenAIProvider``,
``AnthropicProvider``, ``GeminiProvider``) needs an ``httpx.AsyncClient``
with the same basic lifecycle: build (or accept an injected) client on
construction, expose it for making requests, and close it cleanly when the
adapter is done being used. ``HTTPProviderMixin`` factors that boilerplate
into one place so each adapter's ``__init__`` only needs one call
(:meth:`~HTTPProviderMixin._init_http_client`) instead of repeating client
construction, ownership tracking, and async context manager plumbing four
times over.
"""

from __future__ import annotations

import logging
from typing import Self

import httpx

logger = logging.getLogger(__name__)


def require_non_empty_string(value: str, field_name: str) -> str:
    """
    Validate required string constructor arguments.

    Used by provider ``__init__`` methods to fail fast (with a clear error)
    when a required credential like an API key is missing or blank, rather
    than deferring the failure to the first HTTP call, where the resulting
    ``401``/``403`` error would be harder to trace back to a configuration
    mistake.

    Args:
        value: The raw string to validate, e.g. an API key passed to a
            provider constructor.
        field_name: The name of the argument being validated, used to build
            a precise error message (e.g. ``"api_key"``).

    Returns:
        ``value`` with leading/trailing whitespace stripped.

    Raises:
        ValueError: If ``value`` is empty, ``None``-like/falsy, or contains
            only whitespace.

    Example:
        >>> require_non_empty_string("  sk-test  ", "api_key")
        'sk-test'
        >>> require_non_empty_string("", "api_key")
        Traceback (most recent call last):
            ...
        ValueError: api_key must be a non-empty string.
    """
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


class HTTPProviderMixin:
    """
    Reusable ``httpx.AsyncClient`` ownership and async context manager support.

    Concrete providers call :meth:`_init_http_client` from ``__init__`` and inherit
    :meth:`close`, :meth:`__aenter__`, and :meth:`__aexit__` without duplication.

    This is a mixin, not a base class with its own ``__init__`` — it declares
    the instance attributes it expects (``_base_url``, ``_timeout``,
    ``_owns_client``, ``_client``) as class-level type annotations only, and
    relies on the concrete provider class to actually populate them by
    calling :meth:`_init_http_client`. It is always combined with
    :class:`~src.providers.base.BaseProvider` via multiple inheritance, e.g.
    ``class OllamaProvider(HTTPProviderMixin, BaseProvider): ...``.

    Attributes:
        _base_url: The provider's API base URL with any trailing slash
            stripped (e.g. ``"https://api.openai.com/v1"``), set by
            :meth:`_init_http_client`.
        _timeout: The request timeout, in seconds, applied to the owned
            client (only meaningful when this class created the client
            itself; an injected client keeps whatever timeout it was
            constructed with).
        _owns_client: Whether this instance created its own
            ``httpx.AsyncClient`` (``True``) versus received one from the
            caller (``False``). Determines whether :meth:`close` actually
            closes the client — a caller-supplied client is assumed to be
            managed (and closed) by whoever created it, since closing a
            client the caller still intends to use elsewhere would be
            surprising and hard to debug.
        _client: The ``httpx.AsyncClient`` used for all HTTP calls made by
            the concrete provider.

    Example:
        >>> import httpx
        >>> class DemoProvider(HTTPProviderMixin):
        ...     def __init__(self, base_url):
        ...         self._init_http_client(
        ...             base_url=base_url,
        ...             timeout=30.0,
        ...             http_client=None,
        ...             provider_label="DemoProvider",
        ...         )
        >>> provider = DemoProvider("https://example.com/")
        >>> provider._base_url
        'https://example.com'
        >>> provider._owns_client
        True
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
        """
        Set up the HTTP client instance attributes for a concrete provider.

        Intended to be called exactly once, from the concrete provider's
        ``__init__``, after any credential/argument validation (e.g.
        :func:`require_non_empty_string`) has already happened.

        Args:
            base_url: The provider's API base URL, e.g.
                ``"https://api.openai.com/v1"``. Any trailing ``/`` is
                stripped so callers can safely join paths with a leading
                ``/`` (e.g. ``self._client.post("/chat/completions", ...)``)
                without risking a doubled slash.
            timeout: Request timeout in seconds, applied only when this
                method constructs a new client (i.e. when ``http_client`` is
                ``None``). Ignored if a client is injected, since that
                client's own timeout configuration takes precedence.
            http_client: An optional pre-built ``httpx.AsyncClient`` to use
                instead of constructing a new one. Passing a client (most
                commonly in tests, to inject a mocked transport, or when a
                caller wants to share one client/connection pool across
                multiple provider instances) means this mixin will *not*
                close it in :meth:`close` — ownership stays with the caller.
            provider_label: A human-readable label used only in the
                initialization log line (e.g. ``"OllamaProvider"``), so log
                output identifies which adapter was constructed.

        Returns:
            None. Populates ``self._base_url``, ``self._timeout``,
            ``self._owns_client``, and ``self._client``.

        Example:
            >>> import httpx
            >>> class DemoProvider(HTTPProviderMixin):
            ...     def __init__(self, client):
            ...         self._init_http_client(
            ...             base_url="https://example.com",
            ...             timeout=10.0,
            ...             http_client=client,
            ...             provider_label="DemoProvider",
            ...         )
            >>> injected = httpx.AsyncClient()
            >>> provider = DemoProvider(injected)
            >>> provider._owns_client
            False
        """
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
        """
        Close the internally created HTTP client, if any.

        Only closes ``self._client`` when this instance created it itself
        (``self._owns_client`` is ``True``); a client injected by the caller
        is left open, since the caller is responsible for its lifecycle and
        may still be using it elsewhere (e.g. shared across multiple
        provider instances).

        Returns:
            None.

        Example:
            >>> import asyncio
            >>> class DemoProvider(HTTPProviderMixin):
            ...     def __init__(self):
            ...         self._init_http_client(
            ...             base_url="https://example.com",
            ...             timeout=10.0,
            ...             http_client=None,
            ...             provider_label="DemoProvider",
            ...         )
            >>> provider = DemoProvider()
            >>> asyncio.run(provider.close())  # closes the owned client
        """
        if self._owns_client:
            await self._client.aclose()
            logger.info("Closed %s HTTP client", self.__class__.__name__)

    async def __aenter__(self) -> Self:
        """
        Enter the async context manager, returning this instance unchanged.

        Enables the ``async with SomeProvider(...) as provider:`` pattern so
        callers get automatic cleanup via :meth:`__aexit__` without having to
        remember to call :meth:`close` manually.

        Returns:
            This same provider instance (``Self``), unmodified.

        Example:
            >>> import asyncio
            >>> class DemoProvider(HTTPProviderMixin):
            ...     def __init__(self):
            ...         self._init_http_client(
            ...             base_url="https://example.com",
            ...             timeout=10.0,
            ...             http_client=None,
            ...             provider_label="DemoProvider",
            ...         )
            >>> async def demo():
            ...     async with DemoProvider() as provider:
            ...         return provider._owns_client
            >>> asyncio.run(demo())
            True
        """
        return self

    async def __aexit__(self, *_args: object) -> None:
        """
        Exit the async context manager, closing any owned HTTP client.

        Args:
            *_args: The standard ``(exc_type, exc_value, traceback)`` triple
                supplied by the ``async with`` protocol on exit. Unused here
                — this mixin always attempts to close the client the same
                way regardless of whether the block exited normally or via
                an exception, so the arguments are intentionally ignored
                rather than named individually.

        Returns:
            None. Never suppresses an exception raised inside the ``async
            with`` block (implicitly returns ``None``/falsy).

        Example:
            >>> import asyncio
            >>> class DemoProvider(HTTPProviderMixin):
            ...     def __init__(self):
            ...         self._init_http_client(
            ...             base_url="https://example.com",
            ...             timeout=10.0,
            ...             http_client=None,
            ...             provider_label="DemoProvider",
            ...         )
            >>> async def demo():
            ...     async with DemoProvider():
            ...         pass  # client is closed automatically on exit
            >>> asyncio.run(demo())
        """
        await self.close()
