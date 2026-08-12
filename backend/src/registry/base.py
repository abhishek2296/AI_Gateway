"""
Abstract model-registry contract for the gateway catalog.

Phase 6.1 introduced :class:`~src.registry.models.ModelInfo` — the *shape* of
one catalog entry. This module defines *how* those entries are stored and
looked up without committing to a specific backend.

Read vs write surfaces
------------------------
- :class:`~src.registry.read_only.ReadOnlyModelRegistry` — used by routes and
  chat during normal runtime (lookup and list only).
- :class:`CatalogModelRegistry` — extends the read contract with
  ``register``, ``unregister``, and ``clear`` for startup
  (:class:`~src.services.model_catalog_loader.ModelCatalogLoader`) and future
  authenticated admin refresh APIs.

``BaseModelRegistry`` is kept as an alias for ``CatalogModelRegistry`` so
existing imports continue to work.
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime

from src.core.enums import ProviderType
from src.registry.models import ModelInfo
from src.registry.read_only import ReadOnlyModelRegistry


class CatalogModelRegistry(ReadOnlyModelRegistry):
    """
    Writable catalog contract used during startup and catalog refresh jobs.

    Subclasses implement storage plus :meth:`record_refresh` so metrics can
    report when the catalog was last repopulated.
    """

    @abstractmethod
    async def register(self, model: ModelInfo) -> None:
        """
        Add one model to the catalog.

        Raises:
            ModelAlreadyRegisteredError: If ``(provider, name)`` already exists.
        """

    @abstractmethod
    async def unregister(self, provider: ProviderType, name: str) -> None:
        """
        Remove one model from the catalog.

        Raises:
            ModelNotFoundError: If the model is not registered.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Remove every catalog entry (used before a full reload)."""

    @abstractmethod
    def record_refresh(self, refreshed_at: datetime) -> None:
        """
        Store the timestamp of the most recent successful catalog load.

        Called by :class:`~src.services.model_catalog_loader.ModelCatalogLoader`
        after ``load()`` finishes so health endpoints can report freshness.

        Args:
            refreshed_at: Timezone-aware UTC timestamp of the refresh.
        """


# Backward-compatible alias used throughout Phase 6.2–6.8 code and tests.
BaseModelRegistry = CatalogModelRegistry

__all__ = ["BaseModelRegistry", "CatalogModelRegistry"]
