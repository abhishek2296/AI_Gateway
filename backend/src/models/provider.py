"""
ORM entity for an AI provider (Ollama, OpenAI, Anthropic, etc.).

A ``Provider`` row is the top of the gateway's routing hierarchy: every
:class:`~src.models.ai_model.AIModel`, :class:`~src.models.api_key.APIKey`,
:class:`~src.models.chat_session.ChatSession`,
:class:`~src.models.usage_record.UsageRecord`, and
:class:`~src.models.provider_health.ProviderHealth` row belongs to exactly
one provider. This module defines only the ``Provider`` table itself —
per-provider connection settings live in the separate 1:1
:class:`~src.models.provider_configuration.ProviderConfiguration` table so
that "what is this backend and what can it do" (this file) stays independent
from "how do we connect to it right now" (configuration).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String, Text, false, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model import AIModel
    from src.models.api_key import APIKey
    from src.models.chat_session import ChatSession
    from src.models.provider_configuration import ProviderConfiguration
    from src.models.provider_health import ProviderHealth
    from src.models.usage_record import UsageRecord


class Provider(Base, TimestampMixin):
    """
    Registry row for a single LLM backend.

    Maps to the ``providers`` table. Represents one vendor/backend the
    gateway knows how to route requests to — for example one row for
    ``"ollama"``, one for ``"openai"``, one for ``"anthropic"``. This is
    catalog metadata (what the backend is and what it supports); it holds
    no secrets and no live connection state (see
    :class:`~src.models.provider_configuration.ProviderConfiguration` and
    :class:`~src.models.provider_health.ProviderHealth` for those concerns).

    **Capability ownership**

    - Flags on this entity describe what the **provider API** supports
      (streaming, embeddings, vision, function calling, audio) — i.e. is
      this capability available *at all* from this backend.
    - Flags on :class:`~src.models.ai_model.AIModel` describe whether
      **that specific model** supports a feature.

    Example: a provider may support vision overall, while only some of its
    models set ``supports_images=True``.

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        name: Stable machine-readable key, e.g. ``"ollama"``, ``"openai"``,
            ``"anthropic"``. Used as a lookup key elsewhere in the system
            (e.g. config, routing). ``String(100)``, unique, indexed.
        display_name: Human-readable label for admin UIs, e.g. ``"OpenAI"``,
            ``"Azure OpenAI"``. ``String(255)``, required.
        description: Optional free-text description of the provider.
            ``Text``, nullable.
        provider_type: Protocol/family hint used to select the right
            adapter, e.g. ``"ollama"`` or ``"openai_compatible"``.
            ``String(100)``, indexed (not unique — multiple provider rows
            can share a protocol family, e.g. several OpenAI-compatible
            backends).
        base_url: Optional base URL to reach the provider's API (e.g. a
            self-hosted Ollama instance or an OpenAI-compatible gateway).
            ``Text``, nullable — official/hosted providers may rely on
            SDK defaults instead of an explicit URL.
        api_version: Optional API version string some providers require
            (e.g. Azure OpenAI's ``api-version`` query parameter).
            ``String(50)``, nullable.
        is_active: Whether this provider is currently eligible for routing.
            ``Boolean``, defaults to ``True`` at both the ORM and database
            level (``server_default=true()``), indexed for fast filtering
            of the active provider set.
        is_local: Whether this provider runs locally/self-hosted (e.g.
            Ollama) as opposed to a remote hosted API. ``Boolean``,
            defaults to ``False``.
        supports_streaming: Whether the provider API can stream responses
            incrementally. ``Boolean``, defaults to ``False``.
        supports_embeddings: Whether the provider API offers an embeddings
            endpoint. ``Boolean``, defaults to ``False``.
        supports_function_calling: Whether the provider API supports
            tool/function calling. ``Boolean``, defaults to ``False``.
        supports_vision: Whether the provider API can accept image input
            for at least some models. ``Boolean``, defaults to ``False``.
        supports_audio: Whether the provider API supports audio
            input/output. ``Boolean``, defaults to ``False``.
        ai_models: One-to-many to :class:`~src.models.ai_model.AIModel`.
            Eagerly loaded (``lazy="selectin"``, a second ``SELECT ... IN``
            query rather than a JOIN, which avoids row duplication for
            one-to-many collections). ``cascade="all, delete-orphan"``:
            deleting a provider deletes its models, and removing a model
            from this collection deletes that model row.
        configuration: One-to-one to
            :class:`~src.models.provider_configuration.ProviderConfiguration`
            (``uselist=False``). Same cascade behavior as ``ai_models`` —
            configuration has no independent lifecycle apart from its
            provider.
        chat_sessions: One-to-many to
            :class:`~src.models.chat_session.ChatSession`. No delete
            cascade here — a provider used by chat history should not be
            deletable while sessions still reference it (the FK's
            ``ondelete="CASCADE"`` on ``ChatSession.provider_id`` is the
            actual DB-level behavior; this relationship is for ORM
            navigation only and doesn't add its own cascade).
        api_keys: One-to-many to :class:`~src.models.api_key.APIKey`.
            ``cascade="all, delete-orphan"`` — keys are meaningless without
            their provider.
        usage_records: One-to-many to
            :class:`~src.models.usage_record.UsageRecord`. Intentionally
            has no delete cascade: usage/billing history must be retained
            even if a provider is later deprecated (the FK uses
            ``ondelete="RESTRICT"``, which prevents deleting a provider
            that still has usage rows at all).
        health_checks: One-to-many to
            :class:`~src.models.provider_health.ProviderHealth`, ordered by
            ``checked_at`` so the collection is naturally chronological.
            ``cascade="all, delete-orphan"`` — health history is only
            meaningful in the context of its provider.

    Example:
        >>> provider = Provider(
        ...     name="ollama",
        ...     display_name="Ollama (local)",
        ...     provider_type="ollama",
        ...     base_url="http://localhost:11434",
        ...     is_local=True,
        ...     supports_streaming=True,
        ... )
        >>> provider.name
        'ollama'
    """

    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Stable machine key, e.g. "ollama", "openai", "anthropic"
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    # Human-readable label, e.g. "OpenAI", "Azure OpenAI"
    display_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Provider family / protocol hint, e.g. "ollama", "openai_compatible"
    provider_type: Mapped[str] = mapped_column(String(100), index=True)

    base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )
    is_local: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    # Provider-level API capabilities
    supports_streaming: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_embeddings: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_function_calling: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_vision: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_audio: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    # lazy="selectin" issues one extra SELECT ... WHERE provider_id IN (...)
    # instead of a JOIN, which avoids duplicating the parent row per child
    # and works uniformly for both sync and async sessions.
    ai_models: Mapped[list[AIModel]] = relationship(
        "AIModel",
        back_populates="provider",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    configuration: Mapped[ProviderConfiguration | None] = relationship(
        "ProviderConfiguration",
        back_populates="provider",
        lazy="selectin",
        uselist=False,
        cascade="all, delete-orphan",
    )
    chat_sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession",
        back_populates="provider",
        lazy="selectin",
    )
    api_keys: Mapped[list[APIKey]] = relationship(
        "APIKey",
        back_populates="provider",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    # No ORM-level delete cascade: usage/billing history must outlive the
    # provider (backed by ondelete="RESTRICT" on the FK), so a provider
    # with usage records cannot be deleted at all.
    usage_records: Mapped[list[UsageRecord]] = relationship(
        "UsageRecord",
        back_populates="provider",
        lazy="selectin",
    )
    health_checks: Mapped[list[ProviderHealth]] = relationship(
        "ProviderHealth",
        back_populates="provider",
        lazy="selectin",
        cascade="all, delete-orphan",
        order_by="ProviderHealth.checked_at",
    )

    def __repr__(self) -> str:
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string of the form ``<Provider id=1 name='ollama'>``. Only
            the primary key and the stable ``name`` are included (not every
            column) to keep log lines short and avoid accidentally leaking
            sensitive fields from related rows.

        Example:
            >>> Provider(id=1, name="ollama").__repr__()
            "<Provider id=1 name='ollama'>"
        """
        return f"<Provider id={self.id} name={self.name!r}>"
