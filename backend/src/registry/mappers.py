"""
Convert provider-layer and ORM model metadata into registry :class:`ModelInfo`.

The provider adapters already know how to describe models via
``list_models()``. This module is the single translation layer so capability
flags and names are not duplicated in the registry package.
"""

from __future__ import annotations

from dataclasses import replace

from src.models.ai_model import AIModel
from src.models.provider import Provider
from src.providers.base import ModelInfo as ProviderModelInfo
from src.core.enums import ProviderType
from src.registry.models import ModelCapability, ModelInfo


def provider_type_from_name(provider_name: str) -> ProviderType:
    """
    Parse a provider machine key into the registry :class:`ProviderType` enum.

    Args:
        provider_name: Stable provider key, e.g. ``"ollama"``.

    Returns:
        Matching :class:`ProviderType` member.

    Raises:
        ValueError: When ``provider_name`` is not a known provider family.
    """
    return ProviderType(provider_name)


def map_provider_model_to_registry(
    provider: ProviderType,
    provider_model: ProviderModelInfo,
) -> ModelInfo:
    """
    Map a provider adapter catalog entry to a registry :class:`ModelInfo`.

    Provider DTOs use boolean ``supports_*`` flags; registry entries use a
    :class:`ModelCapability` frozenset. Chat-capable adapters always receive
    at least :attr:`ModelCapability.CHAT`.

    Args:
        provider: Backend family that offered this model.
        provider_model: One entry from ``BaseProvider.list_models()``.

    Returns:
        A validated registry catalog record ready for ``register()``.

    Example:
        >>> mapped = map_provider_model_to_registry(
        ...     ProviderType.OLLAMA,
        ...     ProviderModelInfo(id="qwen3:8b", name="qwen3:8b"),
        ... )
        >>> ModelCapability.CHAT in mapped.capabilities
        True
    """
    capabilities: set[ModelCapability] = {ModelCapability.CHAT}

    if provider_model.supports_embeddings or "embed" in provider_model.name.lower():
        capabilities.add(ModelCapability.EMBEDDING)
    if provider_model.supports_vision:
        capabilities.add(ModelCapability.VISION)
    if provider_model.supports_streaming:
        capabilities.add(ModelCapability.STREAMING)
    if provider_model.supports_tools:
        capabilities.add(ModelCapability.FUNCTION_CALLING)

    metadata: dict[str, object] = {"provider_model_id": provider_model.id}
    if provider_model.display_name and provider_model.display_name != provider_model.name:
        metadata["provider_display_name"] = provider_model.display_name

    return ModelInfo(
        name=provider_model.name,
        provider=provider,
        display_name=provider_model.display_name or provider_model.name,
        description="",
        capabilities=frozenset(capabilities),
        context_window=provider_model.context_window,
        max_output_tokens=None,
        enabled=True,
        metadata=metadata,
    )


def map_orm_model_to_registry(provider: Provider, ai_model: AIModel) -> ModelInfo:
    """
    Build a registry entry from database ``AIModel`` / ``Provider`` rows.

    Used when the database has catalog metadata (enabled flag, default model,
    display name) that should overlay or supplement provider ``list_models()``
    results.

    Args:
        provider: Owning provider ORM row.
        ai_model: Model ORM row scoped to ``provider``.

    Returns:
        Registry :class:`ModelInfo` reflecting ORM capability columns.
    """
    capabilities: set[ModelCapability] = {ModelCapability.CHAT}
    if ai_model.supports_streaming:
        capabilities.add(ModelCapability.STREAMING)
    if ai_model.supports_images:
        capabilities.add(ModelCapability.VISION)
    if ai_model.supports_tools:
        capabilities.add(ModelCapability.FUNCTION_CALLING)

    metadata: dict[str, object] = {}
    if ai_model.is_default:
        metadata["is_default"] = True

    return ModelInfo(
        name=ai_model.model_name,
        provider=provider_type_from_name(provider.provider_type),
        display_name=ai_model.display_name,
        description=ai_model.description or "",
        capabilities=frozenset(capabilities),
        context_window=ai_model.context_window,
        max_output_tokens=ai_model.max_output_tokens,
        enabled=ai_model.is_active,
        metadata=metadata,
    )


def merge_registry_model(base: ModelInfo, overlay: ModelInfo) -> ModelInfo:
    """
    Merge two registry entries for the same ``(provider, name)`` key.

    The ``overlay`` (typically from the database) wins for human-facing fields
    and the ``enabled`` flag. Capabilities and token limits prefer overlay
    values when present, otherwise keep ``base`` (from live provider catalog).

    Args:
        base: Model metadata discovered from ``list_models()``.
        overlay: Model metadata from the database overlay pass.

    Returns:
        A new immutable :class:`ModelInfo` combining both sources.
    """
    merged_metadata = dict(base.metadata)
    merged_metadata.update(overlay.metadata)

    return replace(
        base,
        display_name=overlay.display_name or base.display_name,
        description=overlay.description or base.description,
        capabilities=base.capabilities | overlay.capabilities,
        context_window=overlay.context_window or base.context_window,
        max_output_tokens=overlay.max_output_tokens or base.max_output_tokens,
        enabled=overlay.enabled,
        metadata=merged_metadata,
        priority=overlay.priority if overlay.priority is not None else base.priority,
        cost_per_input_token=(
            overlay.cost_per_input_token
            if overlay.cost_per_input_token is not None
            else base.cost_per_input_token
        ),
        cost_per_output_token=(
            overlay.cost_per_output_token
            if overlay.cost_per_output_token is not None
            else base.cost_per_output_token
        ),
        latency_score=overlay.latency_score if overlay.latency_score is not None else base.latency_score,
        quality_score=overlay.quality_score if overlay.quality_score is not None else base.quality_score,
        tags=base.tags | overlay.tags,
    )
