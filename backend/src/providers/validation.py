"""
Model list caching for provider validation.

Cloud provider adapters (``OpenAIProvider``, ``AnthropicProvider``,
``GeminiProvider``) validate a requested model id against the provider's
live catalog (``list_models`` / ``validate_model``), which requires an HTTP
round-trip to the vendor's models endpoint. Doing that on every single chat
request would add unnecessary latency and vendor API load for data that
rarely changes within a short window, so this module provides a small,
process-local, time-boxed cache of each provider's model list — hence the
module's name: its purpose is to make *model validation* fast, even though
the cache mechanism itself (:class:`ModelListCache`) is generic.

This is intentionally simple (in-memory, per-process, no persistence) rather
than a shared cache like Redis: model catalogs are small, provider-scoped,
and do not need to be consistent across multiple gateway processes — a short
TTL keeping each process's view "good enough" is sufficient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.providers.base import ModelInfo


@dataclass(slots=True)
class _ModelCacheEntry:
    """
    One cached provider model list plus its expiration timestamp.

    Internal to this module (leading underscore) — callers interact only
    through :class:`ModelListCache`, never construct or inspect entries
    directly.

    Attributes:
        models: The cached tuple of :class:`~src.providers.base.ModelInfo`
            records for one provider, as last fetched from the vendor.
        expires_at: The ``time.monotonic()`` timestamp after which this
            entry is considered stale and must be refetched. Using
            ``time.monotonic()`` rather than wall-clock time avoids the
            cache being thrown off by system clock adjustments (NTP sync,
            manual clock changes, DST), which matters for a duration-based
            expiry check.

    Example:
        >>> entry = _ModelCacheEntry(models=(), expires_at=time.monotonic() + 60)
        >>> entry.expires_at > time.monotonic()
        True
    """

    models: tuple[ModelInfo, ...]
    expires_at: float


class ModelListCache:
    """
    Short-lived in-memory cache of provider model catalogs.

    Each provider adapter instance owns one ``ModelListCache`` (keyed
    internally by provider name, even though in practice each cache instance
    only ever stores one provider's entry) with a TTL sourced from
    ``Settings.PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS``. Entries expire purely
    by time — there is no explicit invalidation hook, so a genuinely changed
    upstream catalog (a vendor deprecating a model) will only be reflected
    once the TTL lapses and the next call refetches it.

    Attributes:
        _ttl_seconds: How long (in seconds) a cached model list remains
            valid after being set, via :meth:`set`.
        _entries: Internal mapping of provider name -> cached entry. Not
            part of the public interface; callers use :meth:`get`/:meth:`set`
            /:meth:`clear` instead of touching this directly.

    Example:
        >>> from src.providers.base import ModelInfo
        >>> cache = ModelListCache(ttl_seconds=60)
        >>> cache.get("openai") is None
        True
        >>> models = (ModelInfo(id="gpt-4o", name="gpt-4o"),)
        >>> cache.set("openai", models)
        >>> cache.get("openai") == models
        True
    """

    def __init__(self, ttl_seconds: float) -> None:
        """
        Create an empty cache with a fixed time-to-live for every entry.

        Args:
            ttl_seconds: How long, in seconds, a cached model list stays
                valid after being stored. Typically sourced from
                ``Settings.PROVIDER_MODEL_LIST_CACHE_TTL_SECONDS`` so it can
                be tuned via configuration rather than hardcoded per
                provider.
        """
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, _ModelCacheEntry] = {}

    def get(self, provider: str) -> tuple[ModelInfo, ...] | None:
        """
        Return the cached model list for ``provider``, if present and fresh.

        Args:
            provider: The provider name the cache entry was stored under
                (e.g. ``"openai"``).

        Returns:
            The cached tuple of :class:`~src.providers.base.ModelInfo` if an
            entry exists and its TTL has not yet elapsed; ``None`` if there
            is no entry, or if the entry has expired (an expired entry is
            treated identically to a missing one — callers refetch and call
            :meth:`set` again either way, so there's no need to distinguish
            the two cases or proactively evict expired entries here).

        Example:
            >>> cache = ModelListCache(ttl_seconds=0.01)
            >>> cache.set("openai", ())
            >>> import time; time.sleep(0.02)
            >>> cache.get("openai") is None
            True
        """
        entry = self._entries.get(provider)
        if entry is None or entry.expires_at <= time.monotonic():
            return None
        return entry.models

    def set(self, provider: str, models: tuple[ModelInfo, ...]) -> None:
        """
        Store (or replace) the cached model list for ``provider``.

        Args:
            provider: The provider name to key the cache entry under.
            models: The freshly-fetched tuple of
                :class:`~src.providers.base.ModelInfo` to cache. Overwrites
                any existing entry for the same provider, resetting its
                expiry to ``now + ttl_seconds``.

        Returns:
            None. This method only mutates the cache's internal state.

        Example:
            >>> cache = ModelListCache(ttl_seconds=60)
            >>> cache.set("ollama", ())
            >>> cache.get("ollama")
            ()
        """
        self._entries[provider] = _ModelCacheEntry(
            models=models,
            expires_at=time.monotonic() + self._ttl_seconds,
        )

    def clear(self) -> None:
        """
        Remove every cached entry, forcing the next :meth:`get` to miss.

        Primarily useful in tests (to reset cache state between cases
        without waiting for TTL expiry) and for any future admin/ops hook
        that needs to force a refresh of all provider catalogs immediately.

        Returns:
            None.

        Example:
            >>> cache = ModelListCache(ttl_seconds=60)
            >>> cache.set("openai", ())
            >>> cache.clear()
            >>> cache.get("openai") is None
            True
        """
        self._entries.clear()
