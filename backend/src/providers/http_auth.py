"""
Authentication header builders for HTTP provider adapters.

Each cloud vendor (OpenAI, Anthropic, Gemini) authenticates HTTP requests
differently — a bearer token, a custom ``x-api-key`` header, or an API key
passed via a Google-specific header — and each has its own set of
vendor-required headers beyond authentication (organization scoping, API
versioning, etc.). Centralizing that here keeps the per-vendor adapter
modules (``openai.py``, ``anthropic.py``, ``gemini.py``) focused on request/
response shape mapping rather than repeating header-building logic.

These builders are pure functions: given credentials/config, they return a
plain ``dict[str, str]`` ready to pass as ``headers=`` to an ``httpx``
request. They never read from environment variables or global state
themselves — callers (the provider ``__init__`` methods) are responsible for
sourcing the actual secret values (e.g. from ``core.config.get_settings()``).
"""

from __future__ import annotations

from typing import Any


def openai_headers(
    api_key: str,
    *,
    organization: str | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build the HTTP headers required to authenticate against the OpenAI API.

    OpenAI uses standard ``Authorization: Bearer <key>`` bearer-token auth.
    The optional ``OpenAI-Organization`` header scopes the request to a
    specific organization when the API key is a member of more than one
    (otherwise OpenAI infers the organization from the key itself).

    Args:
        api_key: The caller's OpenAI API key (e.g. ``"sk-..."``). Not
            validated here — callers should validate non-emptiness before
            calling this (see ``http_mixin.require_non_empty_string``).
        organization: Optional OpenAI organization ID to scope requests to,
            e.g. ``"org-abc123"``. Omitted from the headers entirely when
            ``None`` or empty, letting OpenAI use the key's default org.
        extra: Optional additional headers to merge in, overriding any of
            the defaults above if the keys collide (e.g. for tests or
            experimental vendor headers not yet modeled explicitly).

    Returns:
        A dict of header name -> value, always including ``Authorization``
        and ``Content-Type``, plus ``OpenAI-Organization`` when provided.

    Example:
        >>> openai_headers("sk-test", organization="org-123")
        {'Authorization': 'Bearer sk-test', 'Content-Type': 'application/json', 'OpenAI-Organization': 'org-123'}
        >>> openai_headers("sk-test")
        {'Authorization': 'Bearer sk-test', 'Content-Type': 'application/json'}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if organization:
        headers["OpenAI-Organization"] = organization
    if extra:
        headers.update(extra)
    return headers


def anthropic_headers(
    api_key: str,
    *,
    api_version: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build the HTTP headers required to authenticate against the Anthropic API.

    Anthropic does not use the ``Authorization`` header; instead it expects
    the raw key in a custom ``x-api-key`` header. It also requires an
    explicit ``anthropic-version`` header on every request — Anthropic uses
    this to pin the request/response schema version so vendor-side API
    evolution doesn't silently break older integrations. The version string
    is sourced from configuration (``Settings.ANTHROPIC_API_VERSION``) rather
    than hardcoded here, so it can be bumped without touching this function.

    Args:
        api_key: The caller's Anthropic API key.
        api_version: The Anthropic API version string to pin requests to,
            e.g. ``"2023-06-01"``. Required (not optional/defaulted here)
            because silently omitting it would let Anthropic pick a default
            version that could change without notice.
        extra: Optional additional headers to merge in, overriding defaults
            on key collision.

    Returns:
        A dict of header name -> value including ``x-api-key``,
        ``anthropic-version``, and ``Content-Type``.

    Example:
        >>> anthropic_headers("sk-ant-test", api_version="2023-06-01")
        {'x-api-key': 'sk-ant-test', 'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}
    """
    headers = {
        "x-api-key": api_key,
        "anthropic-version": api_version,
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def gemini_headers(
    api_key: str,
    *,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build the HTTP headers required to authenticate against the Gemini API.

    Google's Generative Language API accepts the API key via the
    ``x-goog-api-key`` header (as an alternative to passing it as a ``key``
    query parameter, which would risk the key leaking into logs/proxies that
    record URLs). Using the header form keeps the key out of request URLs
    and any URL-based logging.

    Args:
        api_key: The caller's Gemini API key.
        extra: Optional additional headers to merge in, overriding defaults
            on key collision.

    Returns:
        A dict of header name -> value including ``x-goog-api-key`` and
        ``Content-Type``.

    Example:
        >>> gemini_headers("test-key")
        {'x-goog-api-key': 'test-key', 'Content-Type': 'application/json'}
    """
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def merge_headers(base: dict[str, str], extra: dict[str, Any] | None) -> dict[str, str]:
    """
    Merge an optional mapping of extra headers into a base header dict.

    Provided as a small shared utility for call sites that build headers
    incrementally (e.g. adding request-specific headers on top of a
    provider's static auth headers) without duplicating the "skip ``None``
    extra, stringify keys/values" logic at every call site.

    Args:
        base: The starting header dict (e.g. from ``openai_headers``). Not
            mutated in place — a new dict is returned.
        extra: Optional mapping of additional headers to merge in. Values
            that are ``None`` are skipped (treated as "no override"), which
            lets callers pass a dict with some conditionally-``None`` values
            without needing to filter it themselves first. Both keys and
            values are coerced with ``str()`` so callers can pass non-string
            values (e.g. an ``int`` request-id) safely.

    Returns:
        A new dict containing all of ``base``'s entries, overridden/extended
        by any non-``None`` entries from ``extra``. Returns ``base`` itself
        unchanged (same object) when ``extra`` is falsy, since there is
        nothing to merge.

    Example:
        >>> merge_headers({"Content-Type": "application/json"}, {"X-Trace-Id": 42})
        {'Content-Type': 'application/json', 'X-Trace-Id': '42'}
        >>> merge_headers({"Content-Type": "application/json"}, None)
        {'Content-Type': 'application/json'}
        >>> merge_headers({"A": "1"}, {"A": None})
        {'A': '1'}
    """
    if not extra:
        return base
    merged = dict(base)
    for key, value in extra.items():
        if value is not None:
            merged[str(key)] = str(value)
    return merged
