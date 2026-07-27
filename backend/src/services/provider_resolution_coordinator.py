"""Coordinate provider resolution, config building, and caching."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.config import Settings, get_settings
from src.core.provider_resolution_cache import ProviderResolutionCache
from src.repositories.ai_model_repository import AIModelRepository
from src.repositories.api_key_repository import APIKeyRepository
from src.repositories.provider_configuration_repository import ProviderConfigurationRepository
from src.repositories.provider_repository import ProviderRepository
from src.services.provider_config_resolver import ProviderConfigResolver
from src.services.provider_resolver import ProviderResolver
from src.services.resolution_types import ResolvedConfiguration


class ProviderResolutionCoordinator:
    """
    Resolve provider configuration with optional TTL caching.

    Opens a short-lived database session per cache miss so callers remain
    framework-agnostic.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings | None = None,
        cache: ProviderResolutionCache | None = None,
        config_resolver: ProviderConfigResolver | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._session_factory = session_factory
        self._cache = cache or ProviderResolutionCache(
            self._settings.PROVIDER_RESOLUTION_CACHE_TTL_SECONDS,
        )
        self._config_resolver = config_resolver or ProviderConfigResolver(self._settings)

    async def resolve(
        self,
        *,
        request_provider: str | None = None,
        request_model: str | None = None,
    ) -> ResolvedConfiguration:
        cached = self._cache.get(request_provider, request_model)
        if cached is not None:
            return cached

        async with self._session_factory() as session:
            resolver = ProviderResolver(
                provider_repository=ProviderRepository(session),
                model_repository=AIModelRepository(session),
                api_key_repository=APIKeyRepository(session),
                configuration_repository=ProviderConfigurationRepository(session),
                settings=self._settings,
            )
            selection = await resolver.resolve(
                request_provider=request_provider,
                request_model=request_model,
            )

        factory_kwargs = self._config_resolver.build(selection)
        resolved = ResolvedConfiguration(
            provider_name=selection.provider_name,
            model_name=selection.model_name,
            factory_kwargs=factory_kwargs,
            provider_source=selection.provider_source,
            model_source=selection.model_source,
        )
        self._cache.set(request_provider, request_model, resolved)
        return resolved
