"""
ORM entity for provider-specific connection and runtime settings.

Separates "how do we connect to this provider right now" (this module) from
"what is this provider and what does it support" (see
:class:`~src.models.provider.Provider`). Splitting these into two tables
means catalog metadata about a provider can be queried/cached independently
of its (potentially more frequently changing, more sensitive) connection
settings, and keeps the credential env-var reference (``api_key_env``)
scoped to a single, narrowly-purposed table.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.provider import Provider


class ProviderConfiguration(Base, TimestampMixin):
    """
    One-to-one connection settings for a :class:`~src.models.provider.Provider`.

    Maps to the ``provider_configurations`` table. Holds how to actually
    reach and call a provider (endpoint, timeouts, retries, proxy) — as
    distinct from ``Provider`` itself, which describes what the backend is
    and what it supports. Enforced 1:1 via the ``UniqueConstraint`` on
    ``provider_id`` plus ``uselist=False`` on the owning side (see
    :attr:`~src.models.provider.Provider.configuration`).

    ``api_key_env`` stores the **environment variable name** that holds the
    secret (e.g. ``OPENAI_API_KEY``), never the secret value itself, per
    ``08-security.mdc``. Provider-specific knobs that do not warrant
    dedicated columns (e.g. an obscure vendor-specific flag) go in
    ``extra_config`` instead of triggering a schema migration for every
    minor provider quirk.

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        provider_id: Foreign key to ``providers.id``. ``ondelete="CASCADE"``
            — configuration has no meaning without its provider. Indexed,
            and also constrained unique (see ``__table_args__``) to
            guarantee at most one configuration row per provider.
        api_key_env: Name of the environment variable holding this
            provider's default secret (e.g. ``"OPENAI_API_KEY"``), or
            ``None`` if the provider needs no key (e.g. local Ollama) or
            keys are managed exclusively via
            :class:`~src.models.api_key.APIKey` rows instead.
            ``String(255)``, nullable.
        endpoint: Base URL/endpoint to call for this provider, or ``None``
            to use the provider SDK's built-in default. ``Text``,
            nullable.
        organization: Optional provider-side organization id/slug.
            ``String(255)``, nullable.
        region: Optional cloud region hint (relevant for providers like
            AWS Bedrock or Azure OpenAI that are region-scoped).
            ``String(100)``, nullable.
        project: Optional provider-side project id/slug (e.g. GCP project
            for Gemini). ``String(255)``, nullable.
        timeout_seconds: Optional request timeout override for this
            provider, or ``None`` to fall back to the gateway's global
            ``TIMEOUT`` setting. ``Integer``, nullable.
        max_retries: Optional retry-count override for this provider, or
            ``None`` to use the gateway's default retry policy.
            ``Integer``, nullable.
        verify_ssl: Whether TLS certificate verification should be
            enforced for calls to this provider. ``Boolean``, defaults to
            ``True`` at both the ORM and database level
            (``server_default=true()``) — disabling this should be a rare,
            explicit opt-out (e.g. certain self-hosted/dev setups), never
            the default.
        proxy_url: Optional HTTP(S) proxy URL to route requests to this
            provider through, or ``None`` for a direct connection.
            ``Text``, nullable.
        extra_config: Free-form JSON blob for provider-specific settings
            that don't warrant a dedicated column. ``JSONB``, nullable.
        is_active: Whether this configuration is currently in effect.
            ``Boolean``, defaults to ``True``, indexed. Allows disabling a
            configuration (e.g. during maintenance) without deleting it.
        provider: One-to-one back to :class:`~src.models.provider.Provider`.
            Eagerly loaded (``lazy="selectin"``).

    Example:
        >>> config = ProviderConfiguration(
        ...     provider_id=1,
        ...     api_key_env="OPENAI_API_KEY",
        ...     endpoint="https://api.openai.com/v1",
        ...     timeout_seconds=30,
        ...     max_retries=3,
        ... )
        >>> config.api_key_env
        'OPENAI_API_KEY'
    """

    __tablename__ = "provider_configurations"
    __table_args__ = (
        UniqueConstraint("provider_id", name="uq_provider_configurations_provider_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )

    api_key_env: Mapped[str | None] = mapped_column(String(255), nullable=True)
    endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    organization: Mapped[str | None] = mapped_column(String(255), nullable=True)
    region: Mapped[str | None] = mapped_column(String(100), nullable=True)
    project: Mapped[str | None] = mapped_column(String(255), nullable=True)

    timeout_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_retries: Mapped[int | None] = mapped_column(Integer, nullable=True)

    verify_ssl: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
    )
    proxy_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    extra_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="configuration",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<ProviderConfiguration id=1 provider_id=1>``.
            Connection details (endpoint, proxy, env var name) are
            intentionally omitted to keep log output short and to avoid
            hinting at credential locations in log aggregators.

        Example:
            >>> ProviderConfiguration(id=1, provider_id=1).__repr__()
            '<ProviderConfiguration id=1 provider_id=1>'
        """
        return (
            f"<ProviderConfiguration id={self.id} provider_id={self.provider_id}>"
        )
