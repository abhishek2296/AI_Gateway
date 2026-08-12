"""
Computed catalog statistics for the model registry.

Metrics are derived from the live registry contents rather than maintained
counters so values stay correct after a catalog refresh or clear/reload.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RegistryMetrics:
    """
    Snapshot of catalog size and freshness at one point in time.

    Attributes:
        registered_models: Total models currently stored in the registry.
        enabled_models: Models with ``enabled=True`` (available for routing).
        disabled_models: Models present but not routable.
        providers_count: Number of distinct provider families with ≥1 model.
        default_model: Resolved default model name for chat, if known.
        last_refresh_time: When :class:`~src.services.model_catalog_loader.ModelCatalogLoader`
            last finished repopulating the catalog (timezone-aware UTC), or
            ``None`` if the registry has never been refreshed.
    """

    registered_models: int
    enabled_models: int
    disabled_models: int
    providers_count: int
    default_model: str | None
    last_refresh_time: datetime | None
