"""Shared types for database-backed provider resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.models.api_key import APIKey
from src.models.provider import Provider
from src.models.provider_configuration import ProviderConfiguration


class ResolutionSource(str, Enum):
    """Where a resolved provider or model value originated."""

    REQUEST = "request"
    DATABASE = "database"
    SETTINGS = "settings"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class ResolvedApiKey:
    """Non-secret API key reference resolved for a provider."""

    env_var: str | None
    key_identifier: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProviderSelection:
    """Outcome of provider and model resolution before factory kwargs are built."""

    provider_name: str
    model_name: str
    provider_source: ResolutionSource
    model_source: ResolutionSource
    provider: Provider | None = None
    configuration: ProviderConfiguration | None = None
    api_key: APIKey | None = None


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    """Fully resolved provider configuration ready for ProviderFactory.create()."""

    provider_name: str
    model_name: str
    factory_kwargs: dict[str, Any]
    provider_source: ResolutionSource
    model_source: ResolutionSource
