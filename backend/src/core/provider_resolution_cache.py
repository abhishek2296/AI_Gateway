"""In-memory TTL cache for resolved provider configuration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.services.resolution_types import ResolvedConfiguration

logger = logging.getLogger(__name__)

CacheKey = tuple[str | None, str | None]


@dataclass(slots=True)
class _CacheEntry:
    value: ResolvedConfiguration
    expires_at: float


class ProviderResolutionCache:
    """
    Short-lived cache for resolved provider configuration.

    Caches configuration snapshots only — never provider adapter instances.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[CacheKey, _CacheEntry] = {}

    def get(
        self,
        provider_hint: str | None,
        model_hint: str | None,
    ) -> ResolvedConfiguration | None:
        """Return a cached configuration when present and not expired."""
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
        """Store a configuration snapshot until TTL expiry."""
        key = (provider_hint, model_hint)
        self._entries[key] = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + self._ttl_seconds,
        )

    def clear(self) -> None:
        """Remove every cached entry (primarily for tests)."""
        self._entries.clear()
