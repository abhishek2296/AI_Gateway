"""
Model registry metadata types and validation.

This module defines the vocabulary used to describe an LLM "model" in the
gateway's catalog:

- ``ProviderType``     — which vendor/backend a model belongs to.
- ``ModelCapability``  — which features a model supports (chat, vision, ...).
- ``ModelInfo``        — an immutable, validated record combining the above
                          with human-readable metadata (name, limits, etc.).

Nothing here talks to a database, an HTTP API, or a provider SDK. This is a
pure, dependency-free metadata layer that other layers (routing, persistence,
provider adapters) can build on top of in later phases.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.registry.exceptions import InvalidModelMetadataError


class ProviderType(str, Enum):
    """
    Known LLM backend family that a catalog entry (``ModelInfo``) belongs to.

    This is a ``str`` subclass, so members compare equal to their plain string
    value and serialize cleanly to JSON (e.g. in API responses or logs)
    without any extra conversion step.

    Members:
        OLLAMA: Local/self-hosted models served via Ollama.
        OPENAI: Models served by OpenAI's API (e.g. gpt-4o).
        ANTHROPIC: Models served by Anthropic's API (e.g. claude-3-5-sonnet).
        GEMINI: Models served by Google's Gemini API.

    Example:
        >>> ProviderType.OPENAI == "openai"
        True
        >>> ProviderType.OPENAI.value
        'openai'
        >>> ProviderType("anthropic")  # parse from a raw string
        <ProviderType.ANTHROPIC: 'anthropic'>
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ModelCapability(str, Enum):
    """
    A single functional capability that a registered model may support.

    A model can advertise zero, one, or many capabilities at the same time
    (stored as a ``frozenset[ModelCapability]`` on ``ModelInfo``). Consumers
    (e.g. a future routing engine) use these flags to decide whether a model
    is eligible for a given request — for instance, only route an
    image-analysis request to a model that has ``ModelCapability.VISION``.

    Members:
        CHAT: Multi-turn conversational text generation.
        COMPLETION: Single-turn / non-chat text completion.
        EMBEDDING: Converts text into vector embeddings.
        VISION: Can accept and reason about image input.
        IMAGE_GENERATION: Can generate images from a text prompt.
        AUDIO: Can accept and/or produce audio (speech-to-text, text-to-speech).
        STREAMING: Supports incremental/streamed responses (e.g. Server-Sent
            Events) instead of only returning a single final response.
        FUNCTION_CALLING: Supports invoking developer-defined tools/functions.

    Example:
        >>> caps = frozenset({ModelCapability.CHAT, ModelCapability.VISION})
        >>> ModelCapability.VISION in caps
        True
        >>> ModelCapability.EMBEDDING in caps
        False
    """

    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    AUDIO = "audio"
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"


def _require_non_empty_string(value: str, field_name: str) -> str:
    """
    Validate that ``value`` is a string containing non-whitespace text.

    This is a small shared helper used by ``ModelInfo.__post_init__`` to
    validate both ``name`` and ``display_name`` with identical rules, so the
    validation logic (and its error messages) stay consistent and are only
    written once.

    Args:
        value: The raw value to check. Expected to be a ``str``, but any type
            can be passed in (e.g. by mistake), which is why we check
            ``isinstance`` rather than assuming the type hint was honored.
        field_name: The name of the field being validated (e.g. ``"name"``).
            Used only to build a clear error message and to populate
            ``InvalidModelMetadataError.field``.

    Returns:
        The trimmed (leading/trailing whitespace removed) string. Trimming
        means callers don't need to worry about accidental whitespace, e.g.
        ``" gpt-4o "`` becomes ``"gpt-4o"``.

    Raises:
        InvalidModelMetadataError: If ``value`` is not a ``str``, or if it is
            empty / only whitespace (e.g. ``""`` or ``"   "``).

    Example:
        >>> _require_non_empty_string("  qwen3:8b  ", "name")
        'qwen3:8b'
        >>> _require_non_empty_string("", "name")
        Traceback (most recent call last):
            ...
        InvalidModelMetadataError: name must be a non-empty string.
    """
    if not isinstance(value, str):
        raise InvalidModelMetadataError(
            f"{field_name} must be a string.",
            field=field_name,
        )
    normalized = value.strip()
    if not normalized:
        raise InvalidModelMetadataError(
            f"{field_name} must be a non-empty string.",
            field=field_name,
        )
    return normalized


def _validate_positive_int(value: int | None, field_name: str) -> None:
    """
    Validate that an optional integer field is either absent or a positive whole number.

    Used for ``context_window`` and ``max_output_tokens``, both of which are
    *optional* (``None`` means "unknown" or "not applicable") but, when
    provided, must represent a real, positive token count — a value of ``0``
    or a negative number would not make sense for a context window size.

    Note ``bool`` is explicitly rejected even though ``bool`` is technically a
    subclass of ``int`` in Python (``True == 1``). Without this check, a
    caller accidentally passing ``context_window=True`` would silently pass
    validation, which would be confusing.

    Args:
        value: The value to validate, or ``None`` to skip validation entirely
            (meaning the field was not supplied).
        field_name: The name of the field being validated (e.g.
            ``"context_window"``), used for error messages.

    Returns:
        None. This function only raises on failure; it does not transform the
        value (unlike ``_require_non_empty_string``, there is nothing to
        normalize for an integer).

    Raises:
        InvalidModelMetadataError: If ``value`` is not ``None`` and is either
            not an ``int`` (e.g. a ``bool`` or a ``str``), or is ``<= 0``.

    Example:
        >>> _validate_positive_int(128_000, "context_window")  # OK, no error
        >>> _validate_positive_int(None, "context_window")      # OK, skipped
        >>> _validate_positive_int(0, "context_window")
        Traceback (most recent call last):
            ...
        InvalidModelMetadataError: context_window must be greater than zero.
    """
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidModelMetadataError(
            f"{field_name} must be an integer.",
            field=field_name,
        )
    if value <= 0:
        raise InvalidModelMetadataError(
            f"{field_name} must be greater than zero.",
            field=field_name,
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """
    Canonical, validated metadata for one model in the gateway's catalog.

    ``ModelInfo`` is the single source of truth for "what is this model and
    what can it do?" — separate from any provider SDK detail or database row.
    Every instance is validated the moment it is constructed (in
    ``__post_init__``), so if you successfully create a ``ModelInfo`` object,
    you can trust its data is well-formed everywhere else in the codebase.

    It is declared ``frozen=True`` (immutable — fields cannot be reassigned
    after creation) and ``slots=True`` (no ``__dict__``, lower memory
    footprint, matches the style used by DTOs in ``providers/base.py``).
    Because it is frozen, "editing" a field means creating a new instance,
    typically via ``dataclasses.replace(model, enabled=False)``.

    Attributes:
        name: The provider-facing model identifier used in API calls, e.g.
            ``"gpt-4o"`` or ``"qwen3:8b"``. Must be a non-empty string;
            leading/trailing whitespace is stripped automatically.
        provider: Which backend family (``ProviderType``) serves this model,
            e.g. ``ProviderType.OPENAI``.
        display_name: A human-friendly name for UIs/logs, e.g. ``"GPT-4o"``.
            Must be a non-empty string; whitespace is stripped automatically.
        description: A free-text explanation of the model (what it's good
            for, notable limits, etc.). Can be an empty string if no
            description is available yet, but must be a ``str``.
        capabilities: The set of features this model supports, as a
            ``frozenset[ModelCapability]`` (immutable, unordered, dedupes
            automatically). An empty ``frozenset()`` is allowed (means "no
            known capabilities yet").
        context_window: Maximum number of input tokens the model can accept
            in one request, e.g. ``128_000``. ``None`` if unknown. If given,
            must be a positive integer.
        max_output_tokens: Maximum number of tokens the model can generate in
            one response, e.g. ``16_384``. ``None`` if unknown. If given,
            must be a positive integer.
        enabled: Whether this model should currently be offered/routable.
            Defaults to ``True``. Set to ``False`` to keep metadata around
            for a deprecated or temporarily disabled model without deleting
            it from the catalog.
        metadata: A free-form mapping for anything not covered by the typed
            fields above — e.g. ``{"family": "gpt-4", "cost_per_1k": 0.005}``.
            Defaults to an empty ``dict`` (never ``None`` after construction).
            Stored internally as a plain ``dict`` copy so mutating the
            original mapping you passed in afterwards does not affect this
            instance.

    Example:
        >>> model = ModelInfo(
        ...     name="gpt-4o",
        ...     provider=ProviderType.OPENAI,
        ...     display_name="GPT-4o",
        ...     description="OpenAI's flagship multimodal model.",
        ...     capabilities=frozenset({
        ...         ModelCapability.CHAT,
        ...         ModelCapability.VISION,
        ...         ModelCapability.STREAMING,
        ...     }),
        ...     context_window=128_000,
        ...     max_output_tokens=16_384,
        ... )
        >>> model.supports(ModelCapability.VISION)
        True
        >>> model.supports(ModelCapability.EMBEDDING)
        False

        Invalid metadata raises immediately at construction time:

        >>> ModelInfo(
        ...     name="",  # empty -> invalid
        ...     provider=ProviderType.OPENAI,
        ...     display_name="GPT-4o",
        ...     description="...",
        ...     capabilities=frozenset({ModelCapability.CHAT}),
        ... )
        Traceback (most recent call last):
            ...
        InvalidModelMetadataError: name must be a non-empty string.
    """

    name: str
    provider: ProviderType
    display_name: str
    description: str
    capabilities: frozenset[ModelCapability]
    context_window: int | None = None
    max_output_tokens: int | None = None
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and normalize every field right after the dataclass is built.

        Dataclasses call ``__post_init__`` automatically once all fields have
        been assigned by the generated ``__init__``. Because ``ModelInfo`` is
        frozen, we cannot simply do ``self.name = ...`` to normalize/replace a
        value — instead we use ``object.__setattr__(self, "name", ...)`` to
        bypass the immutability guard *only* during this one-time setup step.

        What this method checks/normalizes, in order:
            1. ``name``          — non-empty string (trimmed).
            2. ``provider``      — must be a ``ProviderType`` member.
            3. ``display_name``  — non-empty string (trimmed).
            4. ``description``   — must be a ``str`` (empty allowed).
            5. ``capabilities``  — must be a ``frozenset`` containing only
                                    ``ModelCapability`` members.
            6. ``context_window``    — ``None`` or a positive ``int``.
            7. ``max_output_tokens`` — ``None`` or a positive ``int``.
            8. ``enabled``       — must be a ``bool``.
            9. ``metadata``      — ``None`` becomes ``{}``; otherwise must be
                                    a ``Mapping`` and is copied into a plain
                                    ``dict`` so external mutation of the
                                    original object can't leak into this
                                    "immutable" instance.

        Raises:
            InvalidModelMetadataError: On the first field that fails
                validation (validation stops at the first error rather than
                collecting every error, keeping behavior simple and
                predictable).

        Example:
            This runs automatically — you never call it directly:

            >>> ModelInfo(
            ...     name="qwen3:8b",
            ...     provider=ProviderType.OLLAMA,
            ...     display_name="Qwen3 8B",
            ...     description="Local chat model.",
            ...     capabilities=frozenset({ModelCapability.CHAT}),
            ...     context_window=0,  # invalid: must be > 0
            ... )
            Traceback (most recent call last):
                ...
            InvalidModelMetadataError: context_window must be greater than zero.
        """
        object.__setattr__(self, "name", _require_non_empty_string(self.name, "name"))

        if not isinstance(self.provider, ProviderType):
            raise InvalidModelMetadataError(
                "provider must be a ProviderType enum value.",
                field="provider",
            )

        object.__setattr__(
            self,
            "display_name",
            _require_non_empty_string(self.display_name, "display_name"),
        )

        if not isinstance(self.description, str):
            raise InvalidModelMetadataError(
                "description must be a string.",
                field="description",
            )

        if not isinstance(self.capabilities, frozenset):
            raise InvalidModelMetadataError(
                "capabilities must be a frozenset of ModelCapability values.",
                field="capabilities",
            )
        for capability in self.capabilities:
            if not isinstance(capability, ModelCapability):
                raise InvalidModelMetadataError(
                    "capabilities must contain ModelCapability enum values only.",
                    field="capabilities",
                )

        _validate_positive_int(self.context_window, "context_window")
        _validate_positive_int(self.max_output_tokens, "max_output_tokens")

        if not isinstance(self.enabled, bool):
            raise InvalidModelMetadataError(
                "enabled must be a boolean.",
                field="enabled",
            )

        # `metadata` is optional free-form data: treat a missing/None value as
        # "no extra metadata" (empty dict) rather than forcing every caller to
        # pass `metadata={}` explicitly.
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        elif not isinstance(self.metadata, Mapping):
            raise InvalidModelMetadataError(
                "metadata must be a mapping.",
                field="metadata",
            )
        else:
            # Copy into a plain dict so this "frozen" ModelInfo can't be
            # mutated indirectly later by changing the caller's original dict.
            object.__setattr__(self, "metadata", dict(self.metadata))

    def supports(self, capability: ModelCapability) -> bool:
        """
        Check whether this model advertises a given capability.

        This is a convenience wrapper around ``capability in self.capabilities``
        so calling code reads more naturally (e.g. in a future routing
        engine: "only pick models where ``model.supports(ModelCapability.VISION)``").

        Args:
            capability: The capability to check for, e.g.
                ``ModelCapability.STREAMING``.

        Returns:
            ``True`` if ``capability`` is present in this model's
            ``capabilities`` set, ``False`` otherwise.

        Example:
            >>> model = ModelInfo(
            ...     name="claude-3-5-sonnet",
            ...     provider=ProviderType.ANTHROPIC,
            ...     display_name="Claude 3.5 Sonnet",
            ...     description="...",
            ...     capabilities=frozenset({
            ...         ModelCapability.CHAT,
            ...         ModelCapability.FUNCTION_CALLING,
            ...     }),
            ... )
            >>> model.supports(ModelCapability.FUNCTION_CALLING)
            True
            >>> model.supports(ModelCapability.IMAGE_GENERATION)
            False
        """
        return capability in self.capabilities
