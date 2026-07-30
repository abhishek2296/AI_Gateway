"""
ORM entity for reusable prompt templates.

Prompt templates are catalog content — named, versioned prompt text (with
placeholders) that can be reused across many chat sessions/requests rather
than duplicating prompt copy inline everywhere. This module deliberately
has no foreign key to :class:`~src.models.chat_session.ChatSession`; linking
a template to a session (and rendering it with concrete variable values) is
an API/service-layer concern, not a schema-level relationship.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Integer, String, Text, UniqueConstraint, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base
from src.models.mixins import TimestampMixin


class PromptTemplate(Base, TimestampMixin):
    """
    Versioned, reusable prompt template for gateway workloads.

    Maps to the ``prompt_templates`` table. A single logical template (a
    given ``name``) can have multiple ``version`` rows over time — bumping
    the version rather than mutating ``template`` in place preserves the
    exact prompt text that was used historically, which matters for
    reproducing or auditing past AI outputs.

    ``variables`` holds a JSON schema or default values for template
    placeholders (e.g. ``{"topic": {"type": "string", "default": ""}}``),
    interpreted by the service layer when rendering the template — this
    model only stores the data, it does not render it. No FK to
    :class:`~src.models.chat_session.ChatSession` in this phase — linking
    templates to sessions belongs in the API layer.

    Columns:
        id: Surrogate primary key, autoincrementing integer.
        name: Logical template name shared across versions, e.g.
            ``"summarize_v2"``. ``String(255)``, indexed. Combined with
            ``version`` in a unique constraint (see ``__table_args__``) so
            each version of a named template is a distinct, addressable
            row.
        description: Optional free-text explanation of what this template
            is for. ``Text``, nullable.
        template: The actual template text, including any placeholder
            syntax (e.g. ``"Summarize the following: {text}"``). ``Text``,
            required.
        category: Optional grouping label for organizing templates in an
            admin UI (e.g. ``"summarization"``, ``"code-review"``).
            ``String(100)``, nullable, indexed.
        version: Monotonically increasing version number for this
            template ``name``, starting at ``1``. ``Integer``, defaults to
            ``1`` at both the ORM and database level.
        variables: Optional JSON description of expected placeholder
            variables (schema and/or defaults). ``JSONB``, nullable —
            ``None`` means the template has no documented variables (it
            may still contain placeholders; they're just undocumented).
        is_active: Whether this specific version is currently usable.
            ``Boolean``, defaults to ``True``, indexed. Lets older versions
            be retired (hidden from selection) without deleting them, so
            historical references remain resolvable.

    Example:
        >>> template = PromptTemplate(
        ...     name="summarize",
        ...     template="Summarize the following text:\\n\\n{text}",
        ...     category="summarization",
        ...     version=1,
        ...     variables={"text": {"type": "string"}},
        ... )
        >>> template.version
        1
    """

    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint(
            "name",
            "version",
            name="uq_prompt_templates_name_version",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    template: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")

    variables: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        index=True,
    )

    def __repr__(self) -> str:
        """
        Build a concise, debugger/log-friendly representation of this row.

        Returns:
            A string like ``<PromptTemplate id=1 name='summarize'
            version=1>``, identifying both the logical template and which
            version this particular row represents.

        Example:
            >>> PromptTemplate(id=1, name="summarize", version=1).__repr__()
            "<PromptTemplate id=1 name='summarize' version=1>"
        """
        return (
            f"<PromptTemplate id={self.id} name={self.name!r} "
            f"version={self.version}>"
        )
