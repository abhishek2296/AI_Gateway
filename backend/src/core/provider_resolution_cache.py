"""
In-memory TTL cache for resolved provider configuration.

Resolving a provider/model for a chat request (see
``services/provider_resolution_coordinator.py``) can involve database
lookups (provider + configuration + API key rows) that are relatively
expensive to repeat on every single request. This module provides a small,
process-local cache that lets the resolution coordinator skip those lookups
for a short, configurable window (``PROVIDER_RESOLUTION_CACHE_TTL_SECONDS``
in ``core/config.py``) when the same ``(provider_hint, model_hint)`` pair is
requested again shortly after.

This cache is intentionally minimal infrastructure: it knows nothing about
providers, databases, or HTTP — it only stores ``ResolvedConfiguration``
snapshots (already-computed, provider-agnostic results) keyed by the hints
that produced them, and expires them after a fixed time-to-live.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.services.resolution_types import ResolvedConfiguration

logger = logging.getLogger(__name__)

# A cache key is the pair of hints the caller supplied when asking for a
# resolved configuration — e.g. an explicit provider/model from a chat
# request, or (None, None) when relying entirely on defaults. Two requests
# with the same hints are assumed to resolve to the same configuration
# within the TTL window.
CacheKey = tuple[str | None, str | None]


@dataclass(slots=True)
class _CacheEntry:
    """
    Internal record pairing a cached value with its expiry time.

    Private (leading underscore) because it is purely an implementation
    detail of ``ProviderResolutionCache`` — callers never construct or see
    this type directly; they only ever get back the wrapped
    ``ResolvedConfiguration`` via ``ProviderResolutionCache.get()``.

    Attributes:
        value: The resolved configuration snapshot being cached.
        expires_at: A ``time.monotonic()`` timestamp after which this entry
            is considered stale and must be discarded. Monotonic time (not
            wall-clock time) is used so cache expiry is unaffected by system
            clock adjustments (e.g. NTP sync, daylight saving changes).
    """

    value: ResolvedConfiguration
    expires_at: float


class ProviderResolutionCache:
    """
    Short-lived, in-process cache for resolved provider configuration.

    Caches configuration snapshots only — never provider adapter instances
    (e.g. never an ``OllamaService`` or ``OpenAIService`` object) — because
    adapters may hold live HTTP clients or other resources that should not
    outlive a single request/response cycle. Entries expire automatically
    after a fixed TTL rather than being invalidated explicitly, which keeps
    the cache simple at the cost of occasionally serving configuration that
    is up to ``ttl_seconds`` old; this trade-off is acceptable because
    provider/model configuration changes infrequently relative to request
    volume.

    Example:
        >>> from src.services.resolution_types import (
        ...     ResolutionSource,
        ...     ResolvedConfiguration,
        ... )
        >>> cache = ProviderResolutionCache(ttl_seconds=60.0)
        >>> cache.get("ollama", "qwen3:8b") is None
        True
        >>> resolved = ResolvedConfiguration(
        ...     provider_name="ollama",
        ...     model_name="qwen3:8b",
        ...     factory_kwargs={},
        ...     provider_source=ResolutionSource.SETTINGS,
        ...     model_source=ResolutionSource.SETTINGS,
        ... )
        >>> cache.set("ollama", "qwen3:8b", resolved)
        >>> cache.get("ollama", "qwen3:8b") is resolved
        True
    """

    def __init__(self, ttl_seconds: float) -> None:
        """
        Create an empty cache with a fixed time-to-live for every entry.

        Args:
            ttl_seconds: How many seconds a cached entry remains valid after
                being stored via ``set()``. Typically sourced from
                ``Settings.PROVIDER_RESOLUTION_CACHE_TTL_SECONDS`` so the TTL
                is configurable per environment without code changes.
        """
        self._ttl_seconds = ttl_seconds
        self._entries: dict[CacheKey, _CacheEntry] = {}

    def get(
        self,
        provider_hint: str | None,
        model_hint: str | None,
    ) -> ResolvedConfiguration | None:
        """
        Return a cached configuration when present and not expired.

        Expired entries are evicted lazily (deleted the first time they are
        looked up after expiring) rather than via a background sweep, since
        that avoids needing a separate cleanup task for what is expected to
        be a small, short-lived cache.

        Args:
            provider_hint: The provider name hint from the original
                resolution request (e.g. an explicit ``"openai"`` from a
                chat request), or ``None`` if none was given.
            model_hint: The model name hint from the original resolution
                request, or ``None`` if none was given.

        Returns:
            The previously cached ``ResolvedConfiguration`` for this exact
            ``(provider_hint, model_hint)`` pair if it exists and has not yet
            expired; otherwise ``None`` (cache miss or expired entry).

        Example:
            >>> cache = ProviderResolutionCache(ttl_seconds=60.0)
            >>> cache.get(None, None) is None
            True
        """
        key = (provider_hint, model_hint)
        entry = self._entries.get(key)
        if entry is None:
            logger.debug("Provider resolution cache miss (key=%s)", key)
            return None
        if entry.expires_at <= time.monotonic():
            del self._entries[key]
            logger.debug("Provider resolution cache expired (key=%s)", key)
            return None
        logger.info("Provider resolution cache hit (key=%s)", key)
        return entry.value

    def set(
        self,
        provider_hint: str | None,
        model_hint: str | None,
        value: ResolvedConfiguration,
    ) -> None:
        """
        Store a configuration snapshot, valid until ``ttl_seconds`` from now.

        Overwrites any existing entry for the same ``(provider_hint,
        model_hint)`` pair, resetting its expiry — there is no need to merge
        with a prior value since ``ResolvedConfiguration`` is an immutable,
        fully-computed snapshot.

        Args:
            provider_hint: The provider name hint that produced ``value``
                (matches the argument passed to ``get()``), or ``None``.
            model_hint: The model name hint that produced ``value`` (matches
                the argument passed to ``get()``), or ``None``.
            value: The fully resolved configuration to cache.

        Example:
            >>> from src.services.resolution_types import (
            ...     ResolutionSource,
            ...     ResolvedConfiguration,
            ... )
            >>> cache = ProviderResolutionCache(ttl_seconds=60.0)
            >>> resolved = ResolvedConfiguration(
            ...     provider_name="ollama",
            ...     model_name="qwen3:8b",
            ...     factory_kwargs={},
            ...     provider_source=ResolutionSource.SETTINGS,
            ...     model_source=ResolutionSource.SETTINGS,
            ... )
            >>> cache.set("ollama", "qwen3:8b", resolved)
            >>> cache.get("ollama", "qwen3:8b") is resolved
            True
        """
        key = (provider_hint, model_hint)
        self._entries[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + self._ttl_seconds,
        )

    def clear(self) -> None:
        """
        Remove every cached entry immediately, ignoring TTL.

        Primarily useful for tests that need a clean cache between cases
        without waiting for entries to expire naturally, and for scenarios
        where cached configuration is known to be stale (e.g. after an admin
        updates provider configuration and wants changes to take effect
        immediately rather than after the TTL elapses).

        Example:
            >>> cache = ProviderResolutionCache(ttl_seconds=60.0)
            >>> cache.clear()  # safe even when already empty
        """
        self._entries.clear()
