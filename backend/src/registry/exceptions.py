"""Registry-layer exception hierarchy for model metadata validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.registry.models import ProviderType


class RegistryError(Exception):
    """
    Base exception for every error raised by the model registry package.

    All other registry exceptions (e.g. ``InvalidModelMetadataError``) inherit
    from this class. Catch ``RegistryError`` at a service or route boundary if
    you want to handle *any* registry failure generically, without needing to
    know about every specific subclass.

    Example:
        try:
            build_catalog_entry(raw_data)
        except RegistryError as exc:
            logger.exception("Registry operation failed: %s", exc)
    """


class InvalidModelMetadataError(RegistryError):
    """
    Raised when a ``ModelInfo`` (or other registry metadata) fails validation.

    This is raised from ``ModelInfo.__post_init__`` whenever a field does not
    meet the rules described in ``registry/models.py`` — for example, an empty
    ``name``, a ``context_window`` that is zero or negative, or a ``provider``
    that isn't a ``ProviderType`` enum member.

    Attributes:
        field: The name of the offending field (e.g. ``"name"``,
            ``"context_window"``). ``None`` if the error is not tied to a
            single field. Callers (routes, admin tools, etc.) can use this to
            build precise error messages such as "context_window is invalid"
            instead of a generic failure.

    Example:
        # Raised internally when constructing an invalid ModelInfo:
        ModelInfo(
            name="",  # empty string is not allowed
            provider=ProviderType.OPENAI,
            display_name="GPT-4o",
            description="...",
            capabilities=frozenset({ModelCapability.CHAT}),
        )
        # -> InvalidModelMetadataError("name must be a non-empty string.")
        #    with exc.field == "name"

        # Catching it at a higher layer:
        try:
            model = ModelInfo(name="", ...)
        except InvalidModelMetadataError as exc:
            print(f"Invalid field '{exc.field}': {exc}")
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        """
        Create the exception with a human-readable message and the offending field.

        Args:
            message: A human-readable explanation of what went wrong, e.g.
                "context_window must be greater than zero.". This becomes the
                exception's string representation (``str(exc)``).
            field: Optional name of the dataclass field that failed
                validation. Keyword-only so call sites always read clearly,
                e.g. ``InvalidModelMetadataError("bad value", field="name")``
                rather than a positional argument whose meaning is unclear.
        """
        self.field = field
        super().__init__(message)


class ModelNotFoundError(RegistryError):
    """
    Raised when a requested model is not present in the registry.

    Used by :meth:`~src.registry.base.BaseModelRegistry.get` and
    :meth:`~src.registry.base.BaseModelRegistry.unregister` when no catalog
    entry exists for the given ``(provider, name)`` pair.

    Attributes:
        provider: The :class:`~src.registry.models.ProviderType` (or its string
            value) that was looked up. Helps callers build messages like
            "openai/gpt-4o is not in the catalog".
        name: The model name that was looked up.

    Example:
        try:
            model = await registry.get(ProviderType.OPENAI, "unknown-model")
        except ModelNotFoundError as exc:
            print(f"Missing: {exc.provider}/{exc.name}")
    """

    def __init__(
        self,
        message: str,
        *,
        provider: ProviderType | str | None = None,
        name: str | None = None,
    ) -> None:
        """
        Create the exception with lookup context.

        Args:
            message: Human-readable explanation, e.g.
                "Model 'gpt-4o' is not registered for provider 'openai'.".
            provider: The provider that was queried, if known.
            name: The model name that was queried, if known.
        """
        self.provider = provider
        self.name = name
        super().__init__(message)


class ModelDisabledError(RegistryError):
    """
    Raised when a catalog entry exists but ``enabled`` is ``False``.

    Chat and routing must reject disabled models so operators can retire a
    model without deleting its metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: ProviderType | str | None = None,
        name: str | None = None,
    ) -> None:
        self.provider = provider
        self.name = name
        super().__init__(message)


class AmbiguousModelError(RegistryError):
    """
    Raised when a model name matches more than one provider in the registry.

    Example: looking up ``"gpt-4"`` without specifying ``provider`` when both
    OpenAI and a compatible proxy register that name.
    """

    def __init__(self, message: str, *, name: str | None = None) -> None:
        self.name = name
        super().__init__(message)


class ModelAlreadyRegisteredError(RegistryError):
    """
    Raised when :meth:`~src.registry.base.BaseModelRegistry.register` would
    create a duplicate ``(provider, name)`` entry.

    Registries treat ``(provider, name)`` as a unique key so routing and sync
    jobs never have ambiguous catalog entries. Callers that need upsert
    semantics should check :meth:`~src.registry.base.BaseModelRegistry.exists`
    first or use a future dedicated ``upsert`` on a concrete implementation.

    Attributes:
        provider: The conflicting provider.
        name: The conflicting model name.

    Example:
        try:
            await registry.register(existing_model)
        except ModelAlreadyRegisteredError as exc:
            print(f"Duplicate: {exc.provider}/{exc.name}")
    """

    def __init__(
        self,
        message: str,
        *,
        provider: ProviderType | str | None = None,
        name: str | None = None,
    ) -> None:
        """
        Create the exception with the conflicting key.

        Args:
            message: Human-readable explanation of the duplicate.
            provider: The provider that already has this name registered.
            name: The model name that is already registered.
        """
        self.provider = provider
        self.name = name
        super().__init__(message)
