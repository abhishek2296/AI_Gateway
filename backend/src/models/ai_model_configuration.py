"""
ORM entity for per-model generation and tool defaults.

Stores the "factory defaults" for how an
:class:`~src.models.ai_model.AIModel` should be invoked (temperature,
sampling parameters, default system prompt, etc.) when a request doesn't
override them. This is catalog data, not request state — the
service/API layer is responsible for merging these defaults with any
per-request or per-session overrides before calling the provider.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint, false
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base
from src.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from src.models.ai_model import AIModel


class AIModelConfiguration(Base, TimestampMixin):
    """
    One-to-one default generation settings for an
    :class:`~src.models.ai_model.AIModel`.

    Maps to the ``ai_model_configurations`` table. Each row holds the
    default sampling/generation parameters (temperature, top-p, etc.) and
    tool-related defaults for exactly one model — enforced by the
    ``UniqueConstraint`` on ``ai_model_id`` combined with ``uselist=False``
    on the owning side's relationship (see
    :attr:`~src.models.ai_model.AIModel.configuration`).

    These are catalog defaults (temperature, system prompt template, etc.).
    Request-time overrides belong in the API/service layer, not here — this
    table is never mutated mid-request. Provider-specific extras that don't
    warrant a dedicated column go in ``extra_parameters``.

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        ai_model_id: Foreign key to ``ai_models.id``. ``ondelete="CASCADE"``
            — configuration has no meaning without its model. Indexed, and
            also constrained unique (see ``__table_args__``) to guarantee
            at most one configuration row per model.
        temperature: Default sampling temperature, or ``None`` to defer to
            the provider's own default. ``Float``, nullable.
        top_p: Default nucleus-sampling threshold, or ``None``. ``Float``,
            nullable.
        top_k: Default top-k sampling cutoff, or ``None``. ``Integer``,
            nullable. Not all providers support this parameter.
        frequency_penalty: Default frequency penalty, or ``None``.
            ``Float``, nullable.
        presence_penalty: Default presence penalty, or ``None``. ``Float``,
            nullable.
        max_tokens: Default maximum tokens to generate, or ``None`` to let
            the provider decide. ``Integer``, nullable.
        seed: Default deterministic sampling seed, or ``None`` for
            non-deterministic generation. ``Integer``, nullable.
        system_prompt_template: Optional default system prompt (or
            template string) to prepend when a session doesn't supply its
            own. ``Text``, nullable.
        json_mode_default: Whether structured/JSON output mode is enabled
            by default for this model. ``Boolean``, defaults to ``False``.
        tool_choice_default: Default tool-choice policy string (e.g.
            ``"auto"``, ``"none"``, a specific tool name), or ``None`` to
            use the provider's default behavior. ``String(100)``, nullable.
        stream_default: Whether responses should stream by default.
            ``Boolean``, defaults to ``False``.
        extra_parameters: Free-form JSON blob for provider-specific
            parameters that don't have a dedicated column (e.g. Anthropic's
            ``top_k`` variants or OpenAI's ``logit_bias``). ``JSONB``,
            nullable.
        ai_model: One-to-one back to
            :class:`~src.models.ai_model.AIModel`. Eagerly loaded
            (``lazy="selectin"``).

    Example:
        >>> config = AIModelConfiguration(
        ...     ai_model_id=1,
        ...     temperature=0.7,
        ...     max_tokens=2048,
        ...     stream_default=True,
        ... )
        >>> config.temperature
        0.7
    """

    __tablename__ = "ai_model_configurations"
    __table_args__ = (
        UniqueConstraint("ai_model_id", name="uq_ai_model_configurations_ai_model_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    ai_model_id: Mapped[int] = mapped_column(
        ForeignKey("ai_models.id", ondelete="CASCADE"),
        index=True,
    )

    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frequency_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    presence_penalty: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)

    system_prompt_template: Mapped[str | None] = mapped_column(Text, nullable=True)

    json_mode_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )
    tool_choice_default: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stream_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default=false(),
    )

    extra_parameters: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    ai_model: Mapped[AIModel] = relationship(
        "AIModel",
        back_populates="configuration",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<AIModelConfiguration id=1 ai_model_id=1>``.
            Generation parameters are omitted to keep log output short and
            because they carry no identifying value on their own.

        Example:
            >>> AIModelConfiguration(id=1, ai_model_id=1).__repr__()
            '<AIModelConfiguration id=1 ai_model_id=1>'
        """
        return (
            f"<AIModelConfiguration id={self.id} ai_model_id={self.ai_model_id}>"
        )
