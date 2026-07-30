"""
ORM entity for an AI model offered by a provider.

Where :class:`~src.models.provider.Provider` answers "which backend", this
module answers "which model on that backend" — e.g. ``qwen3:8b`` under the
``ollama`` provider, or ``gpt-4o`` under the ``openai`` provider. Each
:class:`~src.models.chat_session.ChatSession` and
:class:`~src.models.usage_record.UsageRecord` references a specific
``AIModel`` row, not just a provider, so routing and billing can be tracked
per model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint, false, text, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model_configuration import AIModelConfiguration
    from src.models.chat_session import ChatSession
    from src.models.provider import Provider
    from src.models.usage_record import UsageRecord


class AIModel(Base, TimestampMixin):
    """
    A concrete model identifier under a :class:`~src.models.provider.Provider`.

    Maps to the ``ai_models`` table. Capability flags here mean: **this
    specific model** supports the feature (e.g. this particular checkpoint
    accepts image input). Provider-level API capabilities (does the backend
    *offer* an endpoint for that feature at all) live on
    :class:`~src.models.provider.Provider`.

    ``model_name`` is the provider-native id (e.g. ``qwen3:8b``, ``gpt-4o``).
    Uniqueness is scoped per provider (via the composite unique constraint)
    so different backends may share names — e.g. two different
    OpenAI-compatible providers could each have a model literally named
    ``"gpt-4o"`` without conflicting. At most one row per provider may set
    ``is_default=True``, enforced by a partial unique index rather than
    application logic, so the invariant holds even for writes that bypass
    the ORM.

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        provider_id: Foreign key to ``providers.id``. ``ondelete="CASCADE"``
            — deleting a provider deletes all of its model rows, since a
            model has no meaning without its backend. Indexed for fast
            "all models for this provider" lookups.
        model_name: Provider-native model identifier, e.g. ``"qwen3:8b"``.
            ``String(255)``, indexed. Not unique on its own — see
            ``__table_args__`` for the ``(provider_id, model_name)``
            composite uniqueness rule.
        display_name: Human-readable label for admin/UI, e.g.
            ``"Qwen3 8B"``. ``String(255)``, required.
        description: Optional free-text description of the model.
            ``Text``, nullable.
        context_window: Maximum input tokens the model accepts, or
            ``None`` if unknown/not applicable. ``Integer``, nullable.
        max_output_tokens: Maximum tokens the model can generate in one
            response, or ``None`` if unknown. ``Integer``, nullable.
        supports_tools: Whether this model can be invoked with tool/function
            definitions. ``Boolean``, defaults to ``False``.
        supports_json: Whether this model supports a constrained/structured
            JSON output mode. ``Boolean``, defaults to ``False``.
        supports_images: Whether this model accepts image input (vision).
            ``Boolean``, defaults to ``False``.
        supports_streaming: Whether this model's responses can be streamed
            incrementally. ``Boolean``, defaults to ``False``.
        is_default: Whether this is the provider's default model, used when
            a request doesn't specify one explicitly. ``Boolean``, defaults
            to ``False``, indexed. At most one ``True`` row per
            ``provider_id`` — enforced by the partial unique index
            ``uq_ai_models_one_default_per_provider`` (see
            ``__table_args__``), not by application code.
        is_active: Whether this model is currently selectable/routable.
            ``Boolean``, defaults to ``True``, indexed. Used to hide
            deprecated models without deleting their history-bearing rows.
        provider: Many-to-one back to :class:`~src.models.provider.Provider`.
            Eagerly loaded (``lazy="selectin"``).
        configuration: One-to-one to
            :class:`~src.models.ai_model_configuration.AIModelConfiguration`
            (``uselist=False``). ``cascade="all, delete-orphan"`` —
            configuration has no lifecycle independent of its model.
        chat_sessions: One-to-many to
            :class:`~src.models.chat_session.ChatSession`. No delete
            cascade defined at the ORM level; the FK itself uses
            ``ondelete="CASCADE"``, so deleting a model also removes any
            sessions that reference it directly in the database.
        usage_records: One-to-many to
            :class:`~src.models.usage_record.UsageRecord`. The FK uses
            ``ondelete="RESTRICT"``, so a model with recorded usage cannot
            be deleted — billing/audit history must be preserved.

    Example:
        >>> model = AIModel(
        ...     provider_id=1,
        ...     model_name="qwen3:8b",
        ...     display_name="Qwen3 8B",
        ...     context_window=32_768,
        ...     supports_tools=True,
        ...     is_default=True,
        ... )
        >>> model.model_name
        'qwen3:8b'
    """

    __tablename__ = "ai_models"
    __table_args__ = (
        # Same model name may recur across different providers; uniqueness
        # only needs to hold within a single provider's catalog.
        UniqueConstraint(
            "provider_id",
            "model_name",
            name="uq_ai_models_provider_id_model_name",
        ),
        # Partial unique index (Postgres-specific WHERE clause) rather than a
        # plain UniqueConstraint: a full unique constraint on
        # (provider_id, is_default) would still allow only one row per
        # provider to have is_default=False mapped uniquely, which is not
        # what we want — we only need to constrain the *True* case to one
        # row per provider, and rows with is_default=False are unrestricted.
        Index(
            "uq_ai_models_one_default_per_provider",
            "provider_id",
            unique=True,
            postgresql_where=text("is_default IS TRUE"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    provider_id: Mapped[int] = mapped_column(
        ForeignKey("providers.id", ondelete="CASCADE"),
        index=True,
    )

    model_name: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Model-level capabilities (this model only)
    supports_tools: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_json: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_images: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    supports_streaming: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )

    provider: Mapped[Provider] = relationship(
        "Provider",
        back_populates="ai_models",
        lazy="selectin",
    )
    configuration: Mapped[AIModelConfiguration | None] = relationship(
        "AIModelConfiguration",
        back_populates="ai_model",
        lazy="selectin",
        uselist=False,
        cascade="all, delete-orphan",
    )
    chat_sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession",
        back_populates="ai_model",
        lazy="selectin",
    )
    usage_records: Mapped[list[UsageRecord]] = relationship(
        "UsageRecord",
        back_populates="ai_model",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<AIModel id=1 model_name='qwen3:8b'
            provider_id=1>``, identifying both the model and its owning
            provider without needing a separate query.

        Example:
            >>> AIModel(id=1, model_name="qwen3:8b", provider_id=1).__repr__()
            "<AIModel id=1 model_name='qwen3:8b' provider_id=1>"
        """
        return (
            f"<AIModel id={self.id} model_name={self.model_name!r} "
            f"provider_id={self.provider_id}>"
        )
