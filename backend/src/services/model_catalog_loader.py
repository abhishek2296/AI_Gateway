"""
Populate the model registry at application startup.

Loads catalog entries from each provider's ``list_models()`` API, overlays
database ``AIModel`` metadata, and seeds settings defaults so registry-only
chat resolution always has at least one entry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

import src.providers.anthropic  # noqa: F401
import src.providers.gemini  # noqa: F401
import src.providers.ollama  # noqa: F401
import src.providers.openai  # noqa: F401

from src.core.config import Settings, get_settings
from src.models.ai_model import AIModel
from src.models.provider import Provider
from src.providers.base import BaseProvider
from src.providers.exceptions import ProviderError
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry, get_registry
from src.registry.base import CatalogModelRegistry
from src.registry.exceptions import ModelAlreadyRegisteredError
from src.registry.mappers import (
    map_orm_model_to_registry,
    map_provider_model_to_registry,
    merge_registry_model,
    provider_type_from_name,
)
from src.core.enums import ProviderType
from src.registry.models import ModelCapability, ModelInfo
from src.services.provider_config_resolver import ProviderConfigResolver
from src.services.resolution_types import ResolutionSource, ResolvedProviderSelection

logger = logging.getLogger(__name__)


class ModelCatalogLoader:
    """
    Best-effort startup job that fills the registry from live provider catalogs.

    Each provider is queried independently — a failure for OpenAI (missing API
    key) must not prevent Ollama models from loading.
    """

    def __init__(
        self,
        *,
        registry: CatalogModelRegistry,
        factory: ProviderFactory,
        config_resolver: ProviderConfigResolver,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings | None = None,
        provider_registry: ProviderRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._factory = factory
        self._config_resolver = config_resolver
        self._session_factory = session_factory
        self._settings = settings or get_settings()
        self._provider_registry = provider_registry or get_registry()

    async def load(self) -> None:
        """Clear and repopulate the registry from providers, DB, and settings."""
        await self._registry.clear()
        await self._load_from_providers()
        await self._overlay_database_models()
        await self._seed_settings_default()
        # Record refresh time so health/metrics can report catalog freshness.
        self._registry.record_refresh(datetime.now(UTC))

    async def _load_from_providers(self) -> None:
        for provider_name in self._provider_registry.available():
            try:
                provider_type = provider_type_from_name(provider_name)
            except ValueError:
                logger.warning("Skipping unknown provider name in registry: %s", provider_name)
                continue

            try:
                adapter = self._create_provider_adapter(provider_name)
                provider_models = await adapter.list_models()
            except (ProviderError, ValueError) as exc:
                logger.warning(
                    "Skipping provider catalog load for %s: %s",
                    provider_name,
                    type(exc).__name__,
                )
                continue

            for provider_model in provider_models:
                registry_model = map_provider_model_to_registry(provider_type, provider_model)
                await self._register_or_replace(registry_model)

            logger.info(
                "Loaded %d models from provider=%s",
                len(provider_models),
                provider_name,
            )

    async def _overlay_database_models(self) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(AIModel).options(selectinload(AIModel.provider)),
            )
            rows = result.scalars().all()

        for ai_model in rows:
            if ai_model.provider is None:
                continue
            overlay = map_orm_model_to_registry(ai_model.provider, ai_model)
            if await self._registry.exists(overlay.provider, overlay.name):
                existing = await self._registry.get(overlay.provider, overlay.name)
                merged = merge_registry_model(existing, overlay)
                await self._register_or_replace(merged)
            else:
                await self._register_or_replace(overlay)

        logger.info("Applied database overlay for %d AIModel rows", len(rows))

    async def _seed_settings_default(self) -> None:
        provider = ProviderType(self._settings.DEFAULT_PROVIDER)
        name = self._settings.DEFAULT_MODEL
        if await self._registry.exists(provider, name):
            return

        fallback = ModelInfo(
            name=name,
            provider=provider,
            display_name=name,
            description="Settings default model (seeded at startup).",
            capabilities=frozenset({ModelCapability.CHAT, ModelCapability.STREAMING}),
            metadata={"is_default": True, "seeded_from_settings": True},
        )
        await self._register_or_replace(fallback)
        logger.info(
            "Seeded settings default model provider=%s model=%s",
            provider.value,
            name,
        )

    def _create_provider_adapter(self, provider_name: str) -> BaseProvider:
        selection = ResolvedProviderSelection(
            provider_name=provider_name,
            model_name=self._settings.DEFAULT_MODEL,
            provider_source=ResolutionSource.SETTINGS,
            model_source=ResolutionSource.SETTINGS,
        )
        kwargs = self._config_resolver.build(selection)
        return self._factory.create(provider_name, **kwargs)

    async def _register_or_replace(self, model: ModelInfo) -> None:
        if await self._registry.exists(model.provider, model.name):
            await self._registry.unregister(model.provider, model.name)
        try:
            await self._registry.register(model)
        except ModelAlreadyRegisteredError:
            await self._registry.unregister(model.provider, model.name)
            await self._registry.register(model)


def create_model_catalog_loader(
    registry: CatalogModelRegistry,
    *,
    factory: ProviderFactory | None = None,
    settings: Settings | None = None,
    session_factory: Callable[[], AsyncSession] | async_sessionmaker[AsyncSession] | None = None,
) -> ModelCatalogLoader:
    """Build a catalog loader with shared gateway defaults."""
    from src.core.database import AsyncSessionLocal

    resolved_settings = settings or get_settings()
    resolved_factory = factory or ProviderFactory(get_registry())
    resolved_session_factory = session_factory or AsyncSessionLocal
    config_resolver = ProviderConfigResolver(resolved_settings)
    return ModelCatalogLoader(
        registry=registry,
        factory=resolved_factory,
        config_resolver=config_resolver,
        session_factory=resolved_session_factory,
        settings=resolved_settings,
    )
