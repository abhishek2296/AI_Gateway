"""Database-backed provider and model resolution for AIService."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from src.core.config import Settings
from src.models.provider import Provider
from src.repositories.ai_model_repository import AIModelRepository
from src.repositories.api_key_repository import APIKeyRepository
from src.repositories.provider_configuration_repository import ProviderConfigurationRepository
from src.repositories.provider_repository import ProviderRepository
from src.services.resolution_types import (
    ResolutionSource,
    ResolvedApiKey,
    ResolvedProviderSelection,
)

logger = logging.getLogger(__name__)

HARDCODED_PROVIDER_FALLBACK = "ollama"

ProviderNameStrategy = Callable[[], Awaitable[str | None]]


class ProviderResolutionError(Exception):
    """Raised when provider or model resolution fails before adapter invocation."""


class ProviderDisabledError(ProviderResolutionError):
    """Raised when a requested provider exists in the database but is inactive."""


class ProviderResolver:
    """
    Resolve provider identity, model name, and credential references.

    Framework-agnostic — safe for CLI tools, workers, and HTTP services.
    Resolution follows an ordered strategy chain rather than nested conditionals.
    """

    def __init__(
        self,
        provider_repository: ProviderRepository,
        model_repository: AIModelRepository,
        api_key_repository: APIKeyRepository,
        configuration_repository: ProviderConfigurationRepository,
        settings: Settings,
    ) -> None:
        self._provider_repository = provider_repository
        self._model_repository = model_repository
        self._api_key_repository = api_key_repository
        self._configuration_repository = configuration_repository
        self._settings = settings
        self._provider_name_strategies: tuple[tuple[ResolutionSource, ProviderNameStrategy], ...] = (
            (ResolutionSource.DATABASE, self._resolve_database_default_provider),
            (ResolutionSource.SETTINGS, self._resolve_settings_provider),
            (ResolutionSource.FALLBACK, self._resolve_hardcoded_provider),
        )

    async def resolve(
        self,
        *,
        request_provider: str | None = None,
        request_model: str | None = None,
    ) -> ResolvedProviderSelection:
        """Resolve provider, model, and related database rows."""
        provider_name, provider_source, provider_row = await self._resolve_provider_name(
            request_provider,
        )
        model_name, model_source = await self._resolve_model_name(
            request_model=request_model,
            provider_row=provider_row,
            provider_name=provider_name,
        )
        configuration = await self._load_configuration(provider_row)
        api_key = await self._load_api_key(provider_row)

        logger.info(
            "Resolved provider=%s (source=%s), model=%s (source=%s)",
            provider_name,
            provider_source.value,
            model_name,
            model_source.value,
        )
        return ResolvedProviderSelection(
            provider_name=provider_name,
            model_name=model_name,
            provider_source=provider_source,
            model_source=model_source,
            provider=provider_row,
            configuration=configuration,
            api_key=api_key,
        )

    async def resolve_api_key(self, provider: Provider | None) -> ResolvedApiKey | None:
        """Return the default active API key reference for a provider, if any."""
        row = await self._load_api_key(provider)
        if row is None:
            return None
        return ResolvedApiKey(env_var=row.api_key_env, key_identifier=row.key_identifier)

    async def _load_api_key(self, provider: Provider | None):
        if provider is None:
            return None
        return await self._api_key_repository.get_default_for_provider(provider.id)

    async def _resolve_provider_name(
        self,
        request_provider: str | None,
    ) -> tuple[str, ResolutionSource, Provider | None]:
        if request_provider:
            row = await self._provider_repository.get_by_name(request_provider)
            if row is not None and not row.is_active:
                logger.warning("Requested provider %r is disabled", request_provider)
                raise ProviderDisabledError(
                    f"Provider {request_provider!r} is disabled.",
                )
            if row is not None:
                logger.info("Using request provider %r from database", row.name)
                return row.name, ResolutionSource.REQUEST, row
            logger.info("Using request provider %r without database row", request_provider)
            return request_provider, ResolutionSource.REQUEST, None

        for source, strategy in self._provider_name_strategies:
            name = await strategy()
            if not name:
                continue
            row = await self._provider_repository.get_by_name(name)
            if row is not None and not row.is_active:
                logger.warning("Skipping disabled provider %r during resolution", name)
                continue
            if row is not None:
                return name, ResolutionSource.DATABASE, row
            return name, source, None

        raise ProviderResolutionError("Unable to resolve a provider.")

    async def _resolve_model_name(
        self,
        *,
        request_model: str | None,
        provider_row: Provider | None,
        provider_name: str,
    ) -> tuple[str, ResolutionSource]:
        if request_model:
            logger.info("Using request model %r", request_model)
            return request_model, ResolutionSource.REQUEST

        if provider_row is not None:
            default_model = await self._model_repository.get_default_model(provider_row.id)
            if default_model is not None:
                logger.info(
                    "Using database default model %r for provider %r",
                    default_model.model_name,
                    provider_name,
                )
                return default_model.model_name, ResolutionSource.DATABASE

        logger.info("Using settings default model %r", self._settings.DEFAULT_MODEL)
        return self._settings.DEFAULT_MODEL, ResolutionSource.SETTINGS

    async def _load_configuration(self, provider: Provider | None):
        if provider is None:
            return None
        return await self._configuration_repository.get_by_provider_id(provider.id)

    async def _resolve_database_default_provider(self) -> str | None:
        row = await self._provider_repository.get_default_provider()
        if row is None:
            return None
        logger.info("Using database default provider %r", row.name)
        return row.name

    async def _resolve_settings_provider(self) -> str | None:
        logger.info("Using settings default provider %r", self._settings.DEFAULT_PROVIDER)
        return self._settings.DEFAULT_PROVIDER

    async def _resolve_hardcoded_provider(self) -> str | None:
        logger.info("Using hardcoded provider fallback %r", HARDCODED_PROVIDER_FALLBACK)
        return HARDCODED_PROVIDER_FALLBACK
